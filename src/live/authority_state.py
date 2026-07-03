"""
src/live/authority_state.py  —  4-level Order Authority State Machine

Authority Levels (ascending):
  SHADOW          → virtual execution only, no broker contact
  BROKER_DRY_RUN  → order_build + validation + immediate cancel
  PAPER_ACK       → signal → broker_ack → cancel before fill
  LIMITED_LIVE    → real order → reconciliation (capital=¥1.8M cap enforced)

Promotion: 30 trading days at current level with ALL integrity metrics = 0.
Stop conditions: ANY occurrence triggers ABORT immediately (fail-closed).

Persistence: runtime/authority_state.json (single-record, overwrite on change)
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)
SCHEMA_VERSION = 1

STATE_FILE = Path("runtime/authority_state.json")


# ──────────────────────────────────────────────────────────────────────
#  Authority Levels
# ──────────────────────────────────────────────────────────────────────

class AuthorityLevel(str, Enum):
    SHADOW         = "SHADOW"
    BROKER_DRY_RUN = "BROKER_DRY_RUN"
    PAPER_ACK      = "PAPER_ACK"
    LIMITED_LIVE   = "LIMITED_LIVE"
    ABORTED        = "ABORTED"


LEVEL_ORDER = [
    AuthorityLevel.SHADOW,
    AuthorityLevel.BROKER_DRY_RUN,
    AuthorityLevel.PAPER_ACK,
    AuthorityLevel.LIMITED_LIVE,
]

PROMOTION_TRADING_DAYS = 30     # minimum days at current level before promotion
PROMOTION_CRITERIA_KEYS = [     # all must be 0 over the period
    "authority_mismatch",
    "reconciliation_error",
    "replay_consistency_fail",
    "duplicate_submit",
    "cash_truth_gap",
    "position_truth_gap",
    "orphan_order",
    "idempotency_failure",
]

# Stop conditions: any single occurrence → ABORTED
STOP_CONDITIONS = [
    "unexpected_position",
    "cash_negative",
    "unknown_state",
    "orphan_order_unresolved",
    "manual_override",
    "reconciliation_failure",
]


# ──────────────────────────────────────────────────────────────────────
#  Authority State Record
# ──────────────────────────────────────────────────────────────────────

@dataclass
class AuthorityState:
    level: str                              # AuthorityLevel value
    days_at_level: int = 0
    total_signals: int = 0
    approved_signals: int = 0
    total_orders: int = 0
    executed_orders: int = 0
    authority_mismatch: int = 0
    reconciliation_error: int = 0
    replay_consistency_fail: int = 0
    duplicate_submit: int = 0
    cash_truth_gap: int = 0
    position_truth_gap: int = 0
    orphan_order: int = 0
    idempotency_failure: int = 0
    stop_condition_fired: str = ""          # "" or stop condition name
    last_updated: str = ""
    promotion_history: List[dict] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    @staticmethod
    def initial() -> "AuthorityState":
        return AuthorityState(
            level=AuthorityLevel.SHADOW.value,
            last_updated=_now_iso(),
        )

    def integrity_all_zero(self) -> bool:
        return all(getattr(self, k, 0) == 0 for k in PROMOTION_CRITERIA_KEYS)

    def promotion_ready(self) -> bool:
        return (self.days_at_level >= PROMOTION_TRADING_DAYS
                and self.integrity_all_zero()
                and not self.stop_condition_fired)

    def current_level(self) -> AuthorityLevel:
        return AuthorityLevel(self.level)


# ──────────────────────────────────────────────────────────────────────
#  State Machine Operations
# ──────────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def load_state(path: Path = STATE_FILE) -> AuthorityState:
    if not path.exists():
        return AuthorityState.initial()
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        ph = d.pop("promotion_history", [])
        s  = AuthorityState(**{k: v for k, v in d.items()
                               if k in {f.name for f in dataclasses.fields(AuthorityState)}})
        s.promotion_history = ph
        return s
    except Exception as e:
        logger.error("[AUTHORITY_STATE] load error: %s — using initial", e)
        return AuthorityState.initial()


def save_state(state: AuthorityState, path: Path = STATE_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(state), ensure_ascii=False, indent=2),
                    encoding="utf-8")


def check_stop_conditions(state: AuthorityState, event: dict) -> Optional[str]:
    """Check if event triggers any stop condition. Returns condition name or None."""
    for cond in STOP_CONDITIONS:
        if event.get(cond, 0) >= 1:
            return cond
    return None


def fire_stop(state: AuthorityState, condition: str) -> AuthorityState:
    """Immediately move to ABORTED. Fail-closed."""
    logger.critical("[AUTHORITY_STATE] STOP: %s → ABORTED", condition)
    updated = dataclasses.replace(
        state,
        level=AuthorityLevel.ABORTED.value,
        stop_condition_fired=condition,
        last_updated=_now_iso(),
    )
    return updated


def advance_day(state: AuthorityState) -> AuthorityState:
    """Called once per trading day. Increments counter; checks promotion eligibility."""
    if state.stop_condition_fired:
        return state
    updated = dataclasses.replace(
        state,
        days_at_level=state.days_at_level + 1,
        last_updated=_now_iso(),
    )
    return updated


def attempt_promotion(state: AuthorityState) -> AuthorityState:
    """
    Promote to next level if criteria met.
    Returns updated state (unchanged if not ready or already at highest).
    """
    if state.stop_condition_fired:
        return state
    if not state.promotion_ready():
        return state

    current = state.current_level()
    try:
        idx = LEVEL_ORDER.index(current)
    except ValueError:
        return state

    if idx + 1 >= len(LEVEL_ORDER):
        logger.info("[AUTHORITY_STATE] Already at highest level: %s", current.value)
        return state

    next_level = LEVEL_ORDER[idx + 1]
    logger.info("[AUTHORITY_STATE] PROMOTE %s → %s (days=%d, integrity_ok=%s)",
                current.value, next_level.value,
                state.days_at_level, state.integrity_all_zero())

    rec = {
        "from": current.value,
        "to":   next_level.value,
        "days": state.days_at_level,
        "ts":   _now_iso(),
        "integrity_snapshot": {k: getattr(state, k, 0) for k in PROMOTION_CRITERIA_KEYS},
    }
    history = list(state.promotion_history) + [rec]

    # Reset counters for new level
    updated = dataclasses.replace(
        state,
        level=next_level.value,
        days_at_level=0,
        authority_mismatch=0,
        reconciliation_error=0,
        replay_consistency_fail=0,
        duplicate_submit=0,
        cash_truth_gap=0,
        position_truth_gap=0,
        orphan_order=0,
        idempotency_failure=0,
        promotion_history=history,
        last_updated=_now_iso(),
    )
    return updated


def record_signal(state: AuthorityState, approved: bool) -> AuthorityState:
    return dataclasses.replace(
        state,
        total_signals=state.total_signals + 1,
        approved_signals=state.approved_signals + (1 if approved else 0),
        last_updated=_now_iso(),
    )


def record_execution(state: AuthorityState, executed: bool) -> AuthorityState:
    return dataclasses.replace(
        state,
        total_orders=state.total_orders + 1,
        executed_orders=state.executed_orders + (1 if executed else 0),
        last_updated=_now_iso(),
    )


def increment_counter(state: AuthorityState, key: str, n: int = 1) -> AuthorityState:
    if not hasattr(state, key):
        logger.warning("[AUTHORITY_STATE] Unknown counter: %s", key)
        return state
    return dataclasses.replace(
        state,
        **{key: getattr(state, key) + n},
        last_updated=_now_iso(),
    )


# ──────────────────────────────────────────────────────────────────────
#  Summary
# ──────────────────────────────────────────────────────────────────────

def authority_summary(state: AuthorityState) -> dict:
    ap = (state.approved_signals / max(1, state.total_signals))
    et = (state.executed_orders  / max(1, state.total_orders))
    return {
        "level":               state.level,
        "days_at_level":       state.days_at_level,
        "authority_precision": round(ap, 4),
        "execution_tracking":  round(et, 4),
        "integrity_all_zero":  state.integrity_all_zero(),
        "promotion_ready":     state.promotion_ready(),
        "stop_condition":      state.stop_condition_fired or None,
        "integrity": {k: getattr(state, k, 0) for k in PROMOTION_CRITERIA_KEYS},
        "promotion_history":   state.promotion_history,
    }
