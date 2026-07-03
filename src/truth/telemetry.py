"""
telemetry.py — Truth layer telemetry (append-only JSONL)

Records one observation per reconciliation cycle.
Never modifies past entries. Used for drift detection and operational audit.

Fields designed for:
  - confidence trend monitoring
  - UNKNOWN order accumulation alerts
  - deployment block frequency
  - orphan detection history
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

JST = timezone(timedelta(hours=9))
TELEMETRY_SCHEMA_VERSION = 1


@dataclass
class TruthTelemetryEntry:
    """One observation appended per reconciliation cycle."""
    timestamp: str
    trading_day: str

    confidence_score: float
    broker_factor: float
    unknown_factor: float
    staleness_factor: float
    orphan_factor: float

    broker_available: bool
    deployment_allowed: bool
    deployment_block_reason: str

    n_unknown: int
    n_pending: int
    n_orphan: int
    stale_orders_promoted: int
    chain_valid: bool

    actual_equity: float
    conservative_deployable: float
    optimistic_deployable: float
    effective_deployable: float
    confirmed_exposure: float
    pending_exposure: float
    uncertain_exposure: float
    orphan_exposure: float
    cash_buffer_amount: float

    schema_version: int = TELEMETRY_SCHEMA_VERSION


def build_telemetry_entry(truth_state) -> TruthTelemetryEntry:  # TruthState
    """Extract telemetry fields from a TruthState."""
    ct  = truth_state.capital_truth
    cs  = truth_state.confidence

    return TruthTelemetryEntry(
        timestamp                = truth_state.reconciled_at,
        trading_day              = truth_state.trading_day,

        confidence_score         = cs.score,
        broker_factor            = cs.broker_factor,
        unknown_factor           = cs.unknown_factor,
        staleness_factor         = cs.staleness_factor,
        orphan_factor            = cs.orphan_factor,

        broker_available         = truth_state.broker_available,
        deployment_allowed       = ct.deployment_allowed,
        deployment_block_reason  = ct.deployment_block_reason,

        n_unknown                = truth_state.n_unknown,
        n_pending                = truth_state.n_pending,
        n_orphan                 = truth_state.n_orphan,
        stale_orders_promoted    = truth_state.stale_orders_promoted,
        chain_valid              = truth_state.chain_valid,

        actual_equity            = ct.actual_equity,
        conservative_deployable  = ct.conservative_deployable,
        optimistic_deployable    = ct.optimistic_deployable,
        effective_deployable     = ct.effective_deployable,
        confirmed_exposure       = ct.confirmed_exposure,
        pending_exposure         = ct.pending_exposure,
        uncertain_exposure       = ct.uncertain_exposure,
        orphan_exposure          = ct.orphan_exposure,
        cash_buffer_amount       = ct.cash_buffer_amount,
    )


def append_telemetry(entry: TruthTelemetryEntry, path: Path) -> None:
    """Append one telemetry entry to JSONL. Never overwrites."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(asdict(entry), sort_keys=True, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_telemetry(path: Path) -> list[TruthTelemetryEntry]:
    """Load all telemetry entries (chronological order)."""
    if not path.exists():
        return []
    entries: list[TruthTelemetryEntry] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    fields = set(TruthTelemetryEntry.__dataclass_fields__)
                    entries.append(TruthTelemetryEntry(**{
                        k: v for k, v in data.items() if k in fields
                    }))
                except Exception as e:
                    logger.warning("truth_telemetry: line %d malformed (%s)", lineno, e)
    except Exception as exc:
        logger.warning("truth_telemetry: load failed (%s)", exc)
    return entries


def load_recent_telemetry(path: Path, n: int = 100) -> list[TruthTelemetryEntry]:
    """Load last N entries (most recent first order)."""
    all_entries = load_telemetry(path)
    return all_entries[-n:] if len(all_entries) > n else all_entries
