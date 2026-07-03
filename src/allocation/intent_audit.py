"""
intent_audit.py — Immutable allocation intent audit logging

Append-only pre-order snapshots for counterfactual replay.
Write failures log to WARNING and NEVER block execution.
Schema-versioned, deterministic serialization.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)
SCHEMA_VERSION: int = 1


@dataclass
class AllocationIntentRecord:
    """Pre-order allocation snapshot. Immutable once written."""
    timestamp: str
    symbol: str
    signal_id: str
    strategy_id: str
    requested_size: int
    final_size: int
    capital_efficiency_score: float
    effective_independent_n: float
    portfolio_concentration: float
    concentration_penalty: float
    truth_confidence: float
    truth_regime: str
    allocation_cap: int
    throttle_applied: bool
    throttle_reason: str
    fallback_mode: bool
    fallback_reason: str
    stale_state_detected: bool
    allocation_state_age_sec: float
    execution_allowed: bool
    schema_version: int = SCHEMA_VERSION


def append_intent_record(record: AllocationIntentRecord, path: Path) -> None:
    """
    Fail-safe: write failure logs WARNING but NEVER raises.
    Execution must never be blocked by audit log I/O.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(asdict(record), ensure_ascii=False, sort_keys=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as exc:
        logger.warning("[INTENT_AUDIT] write failed (%s) — execution continues", exc)


def load_intent_records(path: Path) -> list[AllocationIntentRecord]:
    if not path.exists():
        return []
    records: list[AllocationIntentRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            records.append(AllocationIntentRecord(
                timestamp=data.get("timestamp", ""),
                symbol=data.get("symbol", ""),
                signal_id=data.get("signal_id", ""),
                strategy_id=data.get("strategy_id", ""),
                requested_size=int(data.get("requested_size", 0)),
                final_size=int(data.get("final_size", 0)),
                capital_efficiency_score=float(data.get("capital_efficiency_score", 0.0)),
                effective_independent_n=float(data.get("effective_independent_n", 0.0)),
                portfolio_concentration=float(data.get("portfolio_concentration", 0.0)),
                concentration_penalty=float(data.get("concentration_penalty", 0.0)),
                truth_confidence=float(data.get("truth_confidence", 0.0)),
                truth_regime=data.get("truth_regime", ""),
                allocation_cap=int(data.get("allocation_cap", 0)),
                throttle_applied=bool(data.get("throttle_applied", False)),
                throttle_reason=data.get("throttle_reason", ""),
                fallback_mode=bool(data.get("fallback_mode", False)),
                fallback_reason=data.get("fallback_reason", ""),
                stale_state_detected=bool(data.get("stale_state_detected", False)),
                allocation_state_age_sec=float(data.get("allocation_state_age_sec", 0.0)),
                execution_allowed=bool(data.get("execution_allowed", True)),
            ))
        except Exception as exc:
            logger.warning("intent_record parse error (%s) — skipped", exc)
    return records
