"""
position_truth.py — Portfolio position ground truth

Reconstructs authoritative position state from three possible sources,
in descending confidence order:

  1. broker_api   — live position data from kabuステーション API (highest)
  2. reconciled   — matched against execution journal (medium)
  3. inferred     — derived from intent snapshots alone (lowest, broker unavailable)

Orphan positions: broker reports a position with no matching intent journal entry.
  - Must NOT be auto-resolved (could be manual trade or system error)
  - Reduce safe_deployable capital (treated as consumed)
  - Flagged for operator review

Invariants:
  1. confirmed_lots ≥ 0 for all positions
  2. avg_cost_price > 0 for confirmed positions
  3. total position value ≤ actual_equity (no over-deployment)
  4. Orphan positions are never silently merged into confirmed holdings
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from src.truth.execution_truth import ExecutionRecord, TERMINAL_STATES

logger = logging.getLogger(__name__)
JST = timezone(timedelta(hours=9))

LOT_SIZE_DEFAULT = 100   # Tokyo Stock Exchange standard unit


@dataclass
class PositionRecord:
    """Ground truth for a single symbol's holdings."""
    symbol: str
    sector: str = "不明"
    confirmed_lots: int = 0      # from broker API or reconciled fills
    pending_lots: int = 0        # from SUBMITTED/ACKED/PARTIAL_FILL intents
    uncertain_lots: int = 0      # from UNKNOWN intents
    avg_cost_price: float = 0.0  # weighted average cost per share (JPY)
    lot_size: int = LOT_SIZE_DEFAULT
    source: str = "inferred"     # "broker_api" | "reconciled" | "inferred"
    confidence: float = 0.5      # [0, 1]
    last_reconciled_at: str = ""
    schema_version: int = 1

    @property
    def confirmed_value(self) -> float:
        return self.confirmed_lots * self.avg_cost_price * self.lot_size

    @property
    def max_possible_value(self) -> float:
        """Upper bound: confirmed + pending + uncertain."""
        total = self.confirmed_lots + self.pending_lots + self.uncertain_lots
        return total * self.avg_cost_price * self.lot_size

    @property
    def total_lots(self) -> int:
        return self.confirmed_lots + self.pending_lots + self.uncertain_lots


@dataclass
class OrphanRecord:
    """
    A position reported by the broker with no matching intent journal entry.

    These are NEVER auto-resolved. They require explicit operator action.
    Conservative capital accounting: treats orphan as consumed capital.
    """
    symbol: str
    sector: str = "不明"
    lots: int = 0
    estimated_price: float = 0.0   # last known price
    lot_size: int = LOT_SIZE_DEFAULT
    detected_at: str = ""
    source: str = "broker_api"
    notes: str = ""
    schema_version: int = 1

    @property
    def estimated_value(self) -> float:
        return self.lots * self.estimated_price * self.lot_size


@dataclass
class PositionTruth:
    """Authoritative portfolio position state."""
    positions: list = field(default_factory=list)   # list[PositionRecord]
    orphans:   list = field(default_factory=list)   # list[OrphanRecord]
    generated_at: str = ""
    trading_day: str = ""
    source: str = "inferred"     # worst-case source across positions
    schema_version: int = 1

    @property
    def total_confirmed_value(self) -> float:
        return sum(p.confirmed_value for p in self.positions)

    @property
    def total_orphan_value(self) -> float:
        return sum(o.estimated_value for o in self.orphans)

    @property
    def total_uncertain_value(self) -> float:
        return sum(
            p.uncertain_lots * p.avg_cost_price * p.lot_size
            for p in self.positions
        )

    @property
    def n_confirmed(self) -> int:
        return sum(1 for p in self.positions if p.confirmed_lots > 0)

    @property
    def symbols_held(self) -> list[str]:
        return [p.symbol for p in self.positions if p.confirmed_lots > 0]


# ── Construction from execution records ───────────────────────────────────────

def build_position_from_records(
    symbol: str,
    sector: str,
    records: list[ExecutionRecord],
) -> PositionRecord:
    """
    Reconstruct a PositionRecord from execution history for one symbol.
    Used when broker API is unavailable (inferred source).
    """
    confirmed_lots = 0
    pending_lots   = 0
    uncertain_lots = 0
    total_cost     = 0.0
    total_filled   = 0

    for rec in records:
        if rec.symbol != symbol:
            continue
        if rec.side == "BUY":
            if rec.state in ("FILLED", "RESOLVED_FILLED"):
                confirmed_lots += rec.filled_qty
                total_cost     += rec.filled_qty * (rec.fill_price or rec.estimated_price)
                total_filled   += rec.filled_qty
            elif rec.state == "PARTIAL_FILL":
                confirmed_lots += rec.filled_qty
                pending_lots   += rec.remaining_qty
                total_cost     += rec.filled_qty * (rec.fill_price or rec.estimated_price)
                total_filled   += rec.filled_qty
            elif rec.state in ("SUBMITTED", "ACKED"):
                pending_lots   += rec.remaining_qty or rec.submitted_qty
            elif rec.state == "UNKNOWN":
                uncertain_lots += rec.remaining_qty or rec.submitted_qty
        elif rec.side == "SELL":
            if rec.state in ("FILLED", "RESOLVED_FILLED"):
                confirmed_lots = max(0, confirmed_lots - rec.filled_qty)

    avg_cost = (total_cost / total_filled) if total_filled > 0 else 0.0

    return PositionRecord(
        symbol          = symbol,
        sector          = sector,
        confirmed_lots  = max(0, confirmed_lots),
        pending_lots    = max(0, pending_lots),
        uncertain_lots  = max(0, uncertain_lots),
        avg_cost_price  = round(avg_cost, 2),
        source          = "inferred",
        confidence      = 0.40,
        last_reconciled_at = datetime.now(JST).isoformat(),
    )


def merge_broker_position(
    record: PositionRecord,
    broker_lots: int,
    broker_avg_cost: float,
    broker_market_price: float,
) -> PositionRecord:
    """
    Override confirmed_lots with broker API data (highest confidence).
    Keeps pending/uncertain from intent journal.
    """
    return PositionRecord(
        symbol          = record.symbol,
        sector          = record.sector,
        confirmed_lots  = max(0, broker_lots),
        pending_lots    = record.pending_lots,
        uncertain_lots  = record.uncertain_lots,
        avg_cost_price  = round(broker_avg_cost or broker_market_price or record.avg_cost_price, 2),
        lot_size        = record.lot_size,
        source          = "broker_api",
        confidence      = 0.95,
        last_reconciled_at = datetime.now(JST).isoformat(),
    )


def build_position_truth(
    execution_records: list[ExecutionRecord],
    broker_positions: dict[str, dict],   # {symbol: {lots, avg_cost, market_price, sector}}
    sector_map: dict[str, str],
    trading_day: str = "",
) -> PositionTruth:
    """
    Construct PositionTruth from execution records + broker positions.

    Algorithm:
      1. Group execution records by symbol.
      2. Build inferred position from records for each known symbol.
      3. Overlay broker positions (confirmed_lots → broker_lots).
      4. Identify orphans: broker has a position we have no records for.
    """
    now = datetime.now(JST).isoformat()
    if not trading_day:
        trading_day = datetime.now(JST).date().isoformat()

    # Group records by symbol
    by_symbol: dict[str, list[ExecutionRecord]] = {}
    for rec in execution_records:
        by_symbol.setdefault(rec.symbol, []).append(rec)

    positions: list[PositionRecord] = []
    broker_symbols_seen: set[str] = set()

    # Only iterate symbols with intent records; orphan detection handles broker-only symbols
    for symbol in sorted(by_symbol.keys()):
        sector = sector_map.get(symbol) or (
            broker_positions.get(symbol, {}).get("sector", "不明")
        )
        # Build from records
        recs = by_symbol.get(symbol, [])
        pos = build_position_from_records(symbol, sector, recs)

        # Overlay broker data if available
        broker_data = broker_positions.get(symbol)
        if broker_data:
            broker_symbols_seen.add(symbol)
            pos = merge_broker_position(
                pos,
                broker_lots       = int(broker_data.get("lots", 0)),
                broker_avg_cost   = float(broker_data.get("avg_cost", 0.0)),
                broker_market_price = float(broker_data.get("market_price", 0.0)),
            )

        # Only include if there's something to show
        if pos.total_lots > 0 or pos.confirmed_lots > 0:
            positions.append(pos)

    # Orphan detection: broker has a symbol we have no intent records for
    orphans: list[OrphanRecord] = []
    for symbol, bdata in broker_positions.items():
        if symbol not in by_symbol and int(bdata.get("lots", 0)) > 0:
            orphans.append(OrphanRecord(
                symbol          = symbol,
                sector          = bdata.get("sector", "不明"),
                lots            = int(bdata.get("lots", 0)),
                estimated_price = float(bdata.get("market_price", bdata.get("avg_cost", 0.0))),
                detected_at     = now,
                notes           = "broker_position_without_intent_record",
            ))

    # Worst-case source
    sources = {p.source for p in positions}
    if "inferred" in sources:
        overall_source = "inferred"
    elif "reconciled" in sources:
        overall_source = "reconciled"
    else:
        overall_source = "broker_api" if broker_positions else "inferred"

    return PositionTruth(
        positions    = positions,
        orphans      = orphans,
        generated_at = now,
        trading_day  = trading_day,
        source       = overall_source,
    )


# ── Persistence ───────────────────────────────────────────────────────────────

def save_position_truth(truth: PositionTruth, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    data = {
        "positions":    [asdict(p) for p in truth.positions],
        "orphans":      [asdict(o) for o in truth.orphans],
        "generated_at": truth.generated_at,
        "trading_day":  truth.trading_day,
        "source":       truth.source,
        "schema_version": truth.schema_version,
    }
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_position_truth(path: Path) -> Optional[PositionTruth]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        positions = [PositionRecord(**p) for p in data.get("positions", [])]
        orphans   = [OrphanRecord(**o)   for o in data.get("orphans", [])]
        return PositionTruth(
            positions    = positions,
            orphans      = orphans,
            generated_at = data.get("generated_at", ""),
            trading_day  = data.get("trading_day", ""),
            source       = data.get("source", "inferred"),
        )
    except Exception as exc:
        logger.warning("position_truth load failed (%s)", exc)
        return None
