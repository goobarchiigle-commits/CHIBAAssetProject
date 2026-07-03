"""
src/analytics/missed_alpha_registry.py
Missed Alpha Registry — deterministic telemetry of skipped winner opportunities.

Purpose:
  "なぜ winner を取れなかったか" を deterministic telemetry 化。
  analytics only — execution unchanged.

INVARIANTS:
  - append-only JSONL
  - observation_only: no execution mutation
  - lookahead contamination forbidden

MissedAlphaReason categories:
  GOVERNANCE_FROZEN       — live governance freeze blocked entry
  ELIGIBILITY_GATE_FAILED — upstream eligibility gate rejected symbol
  MAX_POSITIONS_HIT       — MAX_LIVE_POSITIONS already at cap
  MAX_CAPITAL_HIT         — MAX_LIVE_CAPITAL_PCT exceeded
  MAX_DAILY_ORDERS_HIT    — MAX_LIVE_DAILY_ORDERS reached
  ROLLBACK_ACTIVE         — rollback governor blocked entry
  RECONCILIATION_MISMATCH — reconciliation error blocked entry
  LOW_SCORE               — future_leader_score below threshold
  RSR_ZONE_MISMATCH       — rsr_zone not in eligible set
  UNIVERSE_NOT_ACTIVE     — symbol not in active universe on this date
  TIMING_LATENCY          — signal generated after market close
  DUPLICATE_GUARD         — same symbol already in LIVE_ROLLBACK or exit state
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
JST = timezone(timedelta(hours=9))


class MissedAlphaReason(str, Enum):
    GOVERNANCE_FROZEN        = "governance_frozen"
    ELIGIBILITY_GATE_FAILED  = "eligibility_gate_failed"
    MAX_POSITIONS_HIT        = "max_positions_hit"
    MAX_CAPITAL_HIT          = "max_capital_hit"
    MAX_DAILY_ORDERS_HIT     = "max_daily_orders_hit"
    ROLLBACK_ACTIVE          = "rollback_active"
    RECONCILIATION_MISMATCH  = "reconciliation_mismatch"
    LOW_SCORE                = "low_score"
    RSR_ZONE_MISMATCH        = "rsr_zone_mismatch"
    UNIVERSE_NOT_ACTIVE      = "universe_not_active"
    TIMING_LATENCY           = "timing_latency"
    DUPLICATE_GUARD          = "duplicate_guard"
    OTHER                    = "other"


@dataclass
class MissedAlphaRecord:
    """
    One missed opportunity record.
    Written when a signal was blocked and the symbol subsequently showed positive forward return.
    """
    record_date:       str
    symbol:            str
    reason:            MissedAlphaReason
    future_leader_score: Optional[float]   # score at time of miss
    rsr_zone:          Optional[str]
    governance_hash:   Optional[str]       # hash of the governance cycle that blocked entry
    block_details:     str                 # free-form description of specific blocker
    # Forward return fields (populated by enrichment layer)
    fwd_ret_5d:        Optional[float] = None
    fwd_ret_10d:       Optional[float] = None
    fwd_mfe:           Optional[float] = None   # max favorable excursion
    enriched:          bool = False
    timestamp:         str = field(default_factory=lambda: datetime.now(JST).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_date":          self.record_date,
            "symbol":               self.symbol,
            "reason":               self.reason.value if isinstance(self.reason, MissedAlphaReason) else str(self.reason),
            "future_leader_score":  self.future_leader_score,
            "rsr_zone":             self.rsr_zone,
            "governance_hash":      self.governance_hash,
            "block_details":        self.block_details,
            "fwd_ret_5d":           self.fwd_ret_5d,
            "fwd_ret_10d":          self.fwd_ret_10d,
            "fwd_mfe":              self.fwd_mfe,
            "enriched":             self.enriched,
            "timestamp":            self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MissedAlphaRecord":
        reason_val = d.get("reason", "other")
        try:
            reason = MissedAlphaReason(reason_val)
        except ValueError:
            reason = MissedAlphaReason.OTHER
        return cls(
            record_date=str(d.get("record_date", "")),
            symbol=str(d.get("symbol", "")),
            reason=reason,
            future_leader_score=_opt_float(d.get("future_leader_score")),
            rsr_zone=d.get("rsr_zone"),
            governance_hash=d.get("governance_hash"),
            block_details=str(d.get("block_details", "")),
            fwd_ret_5d=_opt_float(d.get("fwd_ret_5d")),
            fwd_ret_10d=_opt_float(d.get("fwd_ret_10d")),
            fwd_mfe=_opt_float(d.get("fwd_mfe")),
            enriched=bool(d.get("enriched", False)),
            timestamp=str(d.get("timestamp", "")),
        )


def _opt_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ── JSONL I/O ─────────────────────────────────────────────────────────────────

def append_missed_alpha(record: MissedAlphaRecord, log_dir: Path) -> None:
    """
    Append one MissedAlphaRecord to dated JSONL.
    Raises IOError on failure (fail-closed for telemetry integrity).
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    date_str = record.record_date.replace("-", "")
    out_path = log_dir / f"missed_alpha_{date_str}.jsonl"
    line = json.dumps(record.to_dict(), ensure_ascii=False) + "\n"
    with out_path.open("a", encoding="utf-8") as f:
        f.write(line)


def load_missed_alpha(log_dir: Path, date_str: Optional[str] = None) -> List[MissedAlphaRecord]:
    """
    Load MissedAlphaRecords from JSONL.
    date_str: YYYYMMDD to load specific day; None loads all.
    """
    if not log_dir.exists():
        return []
    records: List[MissedAlphaRecord] = []
    if date_str:
        paths = [log_dir / f"missed_alpha_{date_str}.jsonl"]
    else:
        paths = sorted(log_dir.glob("missed_alpha_*.jsonl"))
    for p in paths:
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    records.append(MissedAlphaRecord.from_dict(json.loads(line)))
                except Exception as exc:
                    logger.warning("[MISSED_ALPHA] parse failed: %s", exc)
    return records


# ── Registry run hook ─────────────────────────────────────────────────────────

def record_missed_alpha_from_bridge(
    blocked_symbols:    List[str],
    block_reason:       MissedAlphaReason,
    block_details:      str,
    governance_hash:    Optional[str],
    log_dir:            Path,
    evaluation_date:    str,
    scores:             Optional[Dict[str, float]] = None,
    rsr_zones:          Optional[Dict[str, str]] = None,
) -> List[MissedAlphaRecord]:
    """
    Convenience function to record missed alpha for multiple blocked symbols.
    Called from run_live_signal.py governance bridge hook.
    """
    records: List[MissedAlphaRecord] = []
    for sym in blocked_symbols:
        rec = MissedAlphaRecord(
            record_date=evaluation_date,
            symbol=sym,
            reason=block_reason,
            future_leader_score=(scores or {}).get(sym),
            rsr_zone=(rsr_zones or {}).get(sym),
            governance_hash=governance_hash,
            block_details=block_details,
        )
        try:
            append_missed_alpha(rec, log_dir)
            records.append(rec)
        except Exception as exc:
            logger.warning("[MISSED_ALPHA] append failed for %s: %s", sym, exc)
    return records
