"""
src/execution/regime_drift.py
Regime drift invalidation for deferred entries.

Pure functions — no I/O, no queue mutation.

Drift rules (all thresholds inclusive at the boundary → safe side):
  RSR rank deteriorated by > 20   → [DEFERRED_RANK_DECAY]
  Sector rank deteriorated by > 10 → [DEFERRED_RANK_DECAY]
  Market regime changed            → [DEFERRED_REGIME_DROP]

"Deteriorated" means the rank number increased (higher rank = worse quality).
Checks are skipped when the corresponding creation-context field is None.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.execution.deferred_queue import DeferredEntry

logger = logging.getLogger(__name__)

# ── Thresholds ─────────────────────────────────────────────────────────────────
RSR_RANK_DRIFT_LIMIT:    int = 20
SECTOR_RANK_DRIFT_LIMIT: int = 10


# ── Regime enum ───────────────────────────────────────────────────────────────

class MarketRegime(str, Enum):
    """
    Allowed market regime identifiers.
    Inherits from str so values serialise cleanly to/from JSON without special handling.
    """
    RISK_ON         = "risk_on"
    RISK_OFF        = "risk_off"
    COMMODITY       = "commodity"
    DEFENSIVES      = "defensives"
    RATES_UP        = "rates_up"
    TECH_LEADERSHIP = "tech_leadership"

    @classmethod
    def is_valid(cls, value: str | None) -> bool:
        if value is None:
            return False
        return value in {m.value for m in cls}

    @classmethod
    def coerce(cls, value: str | None) -> Optional[MarketRegime]:
        """Return MarketRegime if value is valid, else None."""
        try:
            return cls(value) if value else None
        except ValueError:
            return None


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RegimeDriftResult:
    should_drop: bool
    reason:      str   # "" | "regime_changed" | "rsr_rank_decay" | "sector_rank_decay"
    detail:      str   # human-readable detail for structured logs


# ── Pure check function ───────────────────────────────────────────────────────

def check_regime_drift(
    entry:               DeferredEntry,
    current_rsr_rank:    Optional[int],
    current_sector_rank: Optional[int],
    current_regime:      Optional[str],
) -> RegimeDriftResult:
    """
    Evaluate regime/rank drift for a single deferred entry.

    Skips individual sub-checks when the entry's creation-context field is None
    (backward compatible with entries created before context capture was added).

    Args:
        entry:               deferred entry to evaluate
        current_rsr_rank:    current universe RSR rank for the symbol (1-based, lower=better)
                             None → RSR rank check skipped
        current_sector_rank: current sector rank for the symbol (1-based, lower=better)
                             None → sector rank check skipped
        current_regime:      current MarketRegime value string
                             None → regime check skipped

    Returns:
        RegimeDriftResult(should_drop=False, reason="", detail="") when no drift detected.
    """
    sym = entry.symbol

    # ── 1. Regime change ──────────────────────────────────────────────────────
    creation_regime = entry.market_regime_at_creation
    if creation_regime is not None and current_regime is not None:
        if creation_regime != current_regime:
            detail = (
                f"symbol={sym} entry_id={entry.entry_id} "
                f"regime_at_creation={creation_regime} current_regime={current_regime}"
            )
            logger.info("[DEFERRED_REGIME_DROP] %s", detail)
            return RegimeDriftResult(
                should_drop = True,
                reason      = "regime_changed",
                detail      = detail,
            )

    # ── 2. RSR rank decay ─────────────────────────────────────────────────────
    creation_rsr = entry.rsr_rank_at_creation
    if creation_rsr is not None and current_rsr_rank is not None:
        drift = current_rsr_rank - creation_rsr  # positive = worse
        if drift > RSR_RANK_DRIFT_LIMIT:
            detail = (
                f"symbol={sym} entry_id={entry.entry_id} "
                f"rsr_rank_at_creation={creation_rsr} "
                f"current_rsr_rank={current_rsr_rank} drift={drift:+d} "
                f"limit={RSR_RANK_DRIFT_LIMIT}"
            )
            logger.info("[DEFERRED_RANK_DECAY] rsr_rank %s", detail)
            return RegimeDriftResult(
                should_drop = True,
                reason      = "rsr_rank_decay",
                detail      = detail,
            )

    # ── 3. Sector rank decay ──────────────────────────────────────────────────
    creation_sec = entry.sector_rank_at_creation
    if creation_sec is not None and current_sector_rank is not None:
        drift = current_sector_rank - creation_sec  # positive = worse
        if drift > SECTOR_RANK_DRIFT_LIMIT:
            detail = (
                f"symbol={sym} entry_id={entry.entry_id} "
                f"sector_rank_at_creation={creation_sec} "
                f"current_sector_rank={current_sector_rank} drift={drift:+d} "
                f"limit={SECTOR_RANK_DRIFT_LIMIT}"
            )
            logger.info("[DEFERRED_RANK_DECAY] sector_rank %s", detail)
            return RegimeDriftResult(
                should_drop = True,
                reason      = "sector_rank_decay",
                detail      = detail,
            )

    return RegimeDriftResult(should_drop=False, reason="", detail="")
