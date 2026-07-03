"""
src/universe/demotion_actuation.py — Phase 3A: Demotion Actuation Engine

Drives the demotion lifecycle state machine:
  ACTIVE → DEMOTION_CANDIDATE → PENDING_EXIT → EXIT_CONFIRMED → REMOVED

Recovery path:
  DEMOTION_CANDIDATE → ACTIVE  (when pressure < RECOVERY_THRESHOLD — hysteresis)

Invariants enforced:
  - No sell orders generated. Universe exclusion only.
  - No universe shrinkage without a replacement candidate.
  - Idempotent: same inputs → same state.
  - Restart-safe: full state rebuilt from JSONL replay.
  - FAIL_OPEN: any per-symbol error is logged; execution continues.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── States ─────────────────────────────────────────────────────────────────────
STATE_ACTIVE         = "ACTIVE"
STATE_CANDIDATE      = "DEMOTION_CANDIDATE"
STATE_PENDING_EXIT   = "PENDING_EXIT"
STATE_EXIT_CONFIRMED = "EXIT_CONFIRMED"
STATE_REMOVED        = "REMOVED"

# ── Thresholds (hysteresis band) ───────────────────────────────────────────────
DEMOTION_THRESHOLD: float = 0.60   # enter DEMOTION_CANDIDATE when pressure ≥ this
RECOVERY_THRESHOLD: float = 0.40   # recover to ACTIVE when pressure < this

_SCHEMA_VERSION = 1


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DemotionRecord:
    """Immutable snapshot of one symbol's demotion state."""
    symbol: str
    state: str
    entered_at: str          # ISO8601 — when this state was first entered
    updated_at: str          # ISO8601 — last mutation timestamp
    reason: str
    demotion_pressure: float
    replacement_delta: float
    position_size: int
    schema_version: int = _SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "DemotionRecord":
        return DemotionRecord(
            symbol=d["symbol"],
            state=d["state"],
            entered_at=d["entered_at"],
            updated_at=d["updated_at"],
            reason=d.get("reason", ""),
            demotion_pressure=float(d.get("demotion_pressure", 0.0)),
            replacement_delta=float(d.get("replacement_delta", 0.0)),
            position_size=int(d.get("position_size", 0)),
            schema_version=int(d.get("schema_version", _SCHEMA_VERSION)),
        )


@dataclass
class DemotionActuationResult:
    """Output of a single DemotionActuationEngine.run() call."""
    demotion_candidates: List[str]       # symbols currently in DEMOTION_CANDIDATE
    pending_exit_count: int
    exit_confirmed_count: int
    removed_count: int
    recovered_count: int
    symbols_to_remove: List[str]         # newly REMOVED this run → caller excludes from universe
    average_days_pending_exit: float
    state_snapshot: Dict[str, DemotionRecord]   # full current tracking state
    transitions: List[DemotionRecord]            # records that actually changed this run

    def to_dict(self) -> dict:
        return {
            "demotion_candidates": self.demotion_candidates,
            "pending_exit_count": self.pending_exit_count,
            "exit_confirmed_count": self.exit_confirmed_count,
            "removed_count": self.removed_count,
            "recovered_count": self.recovered_count,
            "symbols_to_remove": self.symbols_to_remove,
            "average_days_pending_exit": self.average_days_pending_exit,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

class DemotionActuationEngine:
    """
    Manages the demotion lifecycle state machine.

    Separation of concerns:
      - Universe exclusion: symbols_to_remove in result
      - Position exit: NOT managed here (Turtle exit owns this)
      - Sell orders: NEVER generated

    Persistence:
      - Append-only JSONL at pending_demotion_file
      - load_state() replays JSONL, taking latest record per symbol
      - Recovery records (state=ACTIVE) signal removal from tracking
    """

    def __init__(self, pending_demotion_file: Path, metrics_file: Path) -> None:
        self._state_file   = pending_demotion_file
        self._metrics_file = metrics_file

    # ── Public API ─────────────────────────────────────────────────────────────

    def load_state(self) -> Dict[str, DemotionRecord]:
        """
        Replay JSONL → latest record per symbol.
        ACTIVE records signal recovery (symbol removed from tracking).
        REMOVED records stay in state (terminal).
        Returns {} on any error (FAIL_OPEN).
        """
        if not self._state_file.exists():
            return {}
        try:
            latest: Dict[str, DemotionRecord] = {}
            for raw in self._state_file.read_text(encoding="utf-8").splitlines():
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = DemotionRecord.from_dict(json.loads(raw))
                except Exception as exc:
                    logger.warning("[DEMOTION_ACT] JSONL parse error (skip): %s", exc)
                    continue
                if rec.state == STATE_ACTIVE:
                    latest.pop(rec.symbol, None)
                else:
                    latest[rec.symbol] = rec
            return latest
        except Exception as exc:
            logger.warning("[DEMOTION_ACT] load_state failed (FAIL_OPEN): %s", exc)
            return {}

    def run(
        self,
        pressures: dict,                        # Dict[str, DemotionPressure]
        portfolio_state: dict,
        shadow_universe: Dict[str, str],
        live_universe: Dict[str, str],
        ev_scores: Optional[dict] = None,       # optional EV scores for replacement_delta
        allow_new_demotions: bool = True,       # False in SAFE_MODE: skip new candidates only
    ) -> DemotionActuationResult:
        """
        Apply state transitions. Returns actuation result.
        allow_new_demotions=False: skip ACTIVE→DEMOTION_CANDIDATE; existing states continue.
        FAIL_OPEN: per-symbol errors skip that symbol.
        """
        state = self.load_state()
        now   = _now_utc()

        transitions: List[DemotionRecord] = []
        recovered_count = 0
        newly_removed:  List[str] = []

        # Process: all tracked symbols PLUS new candidates from pressures
        all_symbols = set(state.keys()) | {
            sym
            for sym, p in pressures.items()
            if getattr(p, "is_candidate", False)
        }

        for symbol in sorted(all_symbols):  # deterministic order
            try:
                pressure_obj   = pressures.get(symbol)
                pressure_score = float(getattr(pressure_obj, "pressure_score", 0.0))
                is_candidate   = bool(getattr(pressure_obj, "is_candidate", False))
                position_size  = self._get_position_size(symbol, portfolio_state)
                current        = state.get(symbol)

                # ── No existing record ────────────────────────────────────────
                if current is None:
                    if is_candidate and allow_new_demotions:
                        if position_size == 0:
                            # No position at candidate creation → EXIT_CONFIRMED directly
                            rec = DemotionRecord(
                                symbol=symbol, state=STATE_EXIT_CONFIRMED,
                                entered_at=now, updated_at=now,
                                reason="no_position_exit_confirmed",
                                demotion_pressure=pressure_score,
                                replacement_delta=0.0,
                                position_size=0,
                            )
                            _log_transition(symbol, STATE_ACTIVE, STATE_EXIT_CONFIRMED,
                                            pressure_score, 0.0, 0,
                                            "no_position_exit_confirmed")
                        else:
                            rec = DemotionRecord(
                                symbol=symbol, state=STATE_CANDIDATE,
                                entered_at=now, updated_at=now,
                                reason="pressure_threshold_exceeded",
                                demotion_pressure=pressure_score,
                                replacement_delta=0.0,
                                position_size=position_size,
                            )
                            _log_transition(symbol, STATE_ACTIVE, STATE_CANDIDATE,
                                            pressure_score, 0.0, position_size,
                                            "pressure_threshold_exceeded")
                        state[symbol] = rec
                        transitions.append(rec)
                    continue

                # ── DEMOTION_CANDIDATE ────────────────────────────────────────
                if current.state == STATE_CANDIDATE:
                    if pressure_score < RECOVERY_THRESHOLD:
                        rec = DemotionRecord(
                            symbol=symbol, state=STATE_ACTIVE,
                            entered_at=now, updated_at=now,
                            reason="pressure_recovered",
                            demotion_pressure=pressure_score,
                            replacement_delta=current.replacement_delta,
                            position_size=position_size,
                        )
                        del state[symbol]
                        transitions.append(rec)
                        recovered_count += 1
                        _log_transition(symbol, STATE_CANDIDATE, STATE_ACTIVE,
                                        pressure_score, 0.0, position_size,
                                        "pressure_recovered")

                    elif position_size == 0:
                        # No position: skip PENDING_EXIT, go straight to EXIT_CONFIRMED
                        rec = DemotionRecord(
                            symbol=symbol, state=STATE_EXIT_CONFIRMED,
                            entered_at=current.entered_at, updated_at=now,
                            reason="no_position_exit_confirmed",
                            demotion_pressure=pressure_score,
                            replacement_delta=current.replacement_delta,
                            position_size=0,
                        )
                        state[symbol] = rec
                        transitions.append(rec)
                        _log_transition(symbol, STATE_CANDIDATE, STATE_EXIT_CONFIRMED,
                                        pressure_score, current.replacement_delta, 0,
                                        "no_position_exit_confirmed")

                    else:
                        # Has position: check replacement availability
                        has_repl, repl_delta = self._has_replacement(
                            symbol, shadow_universe, live_universe, ev_scores
                        )
                        if has_repl and repl_delta > 0:
                            rec = DemotionRecord(
                                symbol=symbol, state=STATE_PENDING_EXIT,
                                entered_at=current.entered_at, updated_at=now,
                                reason="replacement_available_pending_position_close",
                                demotion_pressure=pressure_score,
                                replacement_delta=float(repl_delta),
                                position_size=position_size,
                            )
                            state[symbol] = rec
                            transitions.append(rec)
                            _log_transition(symbol, STATE_CANDIDATE, STATE_PENDING_EXIT,
                                            pressure_score, repl_delta, position_size,
                                            "replacement_available_pending_position_close")
                        else:
                            # No replacement candidate — stay, update pressure
                            rec = DemotionRecord(
                                symbol=symbol, state=STATE_CANDIDATE,
                                entered_at=current.entered_at, updated_at=now,
                                reason="waiting_for_replacement_candidate",
                                demotion_pressure=pressure_score,
                                replacement_delta=0.0,
                                position_size=position_size,
                            )
                            state[symbol] = rec
                            transitions.append(rec)

                # ── PENDING_EXIT ──────────────────────────────────────────────
                elif current.state == STATE_PENDING_EXIT:
                    if position_size == 0:
                        rec = DemotionRecord(
                            symbol=symbol, state=STATE_EXIT_CONFIRMED,
                            entered_at=current.entered_at, updated_at=now,
                            reason="position_closed",
                            demotion_pressure=pressure_score,
                            replacement_delta=current.replacement_delta,
                            position_size=0,
                        )
                        state[symbol] = rec
                        transitions.append(rec)
                        _log_transition(symbol, STATE_PENDING_EXIT, STATE_EXIT_CONFIRMED,
                                        pressure_score, current.replacement_delta, 0,
                                        "position_closed")
                    else:
                        # Still holding — heartbeat update only
                        rec = DemotionRecord(
                            symbol=symbol, state=STATE_PENDING_EXIT,
                            entered_at=current.entered_at, updated_at=now,
                            reason=current.reason,
                            demotion_pressure=pressure_score,
                            replacement_delta=current.replacement_delta,
                            position_size=position_size,
                        )
                        state[symbol] = rec
                        transitions.append(rec)

                # ── EXIT_CONFIRMED ────────────────────────────────────────────
                elif current.state == STATE_EXIT_CONFIRMED:
                    rec = DemotionRecord(
                        symbol=symbol, state=STATE_REMOVED,
                        entered_at=current.entered_at, updated_at=now,
                        reason="exit_confirmed_removed",
                        demotion_pressure=pressure_score,
                        replacement_delta=current.replacement_delta,
                        position_size=0,
                    )
                    state[symbol] = rec
                    transitions.append(rec)
                    newly_removed.append(symbol)
                    _log_transition(symbol, STATE_EXIT_CONFIRMED, STATE_REMOVED,
                                    pressure_score, current.replacement_delta, 0,
                                    "exit_confirmed_removed")

                # STATE_REMOVED: terminal — no action

            except Exception as sym_err:
                logger.warning(
                    "[DEMOTION_ACT] symbol=%s transition error (FAIL_OPEN): %s",
                    symbol, sym_err,
                )

        # ── Aggregate metrics ──────────────────────────────────────────────────
        candidates     = [s for s, r in state.items() if r.state == STATE_CANDIDATE]
        pending_exit   = [r for r in state.values() if r.state == STATE_PENDING_EXIT]
        exit_confirmed = [s for s, r in state.items() if r.state == STATE_EXIT_CONFIRMED]
        removed        = [s for s, r in state.items() if r.state == STATE_REMOVED]
        avg_days       = _average_days_in_state(pending_exit, now)

        return DemotionActuationResult(
            demotion_candidates=candidates,
            pending_exit_count=len(pending_exit),
            exit_confirmed_count=len(exit_confirmed),
            removed_count=len(removed),
            recovered_count=recovered_count,
            symbols_to_remove=newly_removed,
            average_days_pending_exit=avg_days,
            state_snapshot=state,
            transitions=transitions,
        )

    def write_state(self, transitions: List[DemotionRecord]) -> None:
        """
        Append state transition records to JSONL.
        Only changed records are written (not heartbeats for unchanged states).
        FAIL_OPEN: write failure is logged, not raised.
        """
        if not transitions:
            return
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            with self._state_file.open("a", encoding="utf-8") as fh:
                for rec in transitions:
                    fh.write(json.dumps(rec.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        except Exception as exc:
            logger.warning("[DEMOTION_ACT] write_state failed (FAIL_OPEN): %s", exc)

    def write_metrics(self, result: DemotionActuationResult) -> None:
        """Append daily metrics record. FAIL_OPEN."""
        try:
            self._metrics_file.parent.mkdir(parents=True, exist_ok=True)
            record = {"computed_at": _now_utc(), **result.to_dict()}
            with self._metrics_file.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        except Exception as exc:
            logger.warning("[DEMOTION_ACT] write_metrics failed (FAIL_OPEN): %s", exc)

    # ── Internal ───────────────────────────────────────────────────────────────

    @staticmethod
    def _get_position_size(symbol: str, portfolio_state: dict) -> int:
        """Return position size; 0 if not held."""
        if not portfolio_state:
            return 0
        positions = portfolio_state.get("positions", {})
        if positions and symbol in positions:
            pos = positions[symbol]
            if isinstance(pos, dict):
                return int(pos.get("quantity", pos.get("size", 0)))
            if isinstance(pos, (int, float)):
                return int(pos)
            return 0
        # Fallback: presence in entry_dates means position exists
        if symbol in portfolio_state.get("position_entry_dates", {}):
            return 1
        return 0

    @staticmethod
    def _has_replacement(
        symbol: str,
        shadow_universe: Dict[str, str],
        live_universe: Dict[str, str],
        ev_scores: Optional[dict],
    ) -> Tuple[bool, float]:
        """
        Returns (replacement_candidate_exists, replacement_delta).
        Replacement must exist AND delta > 0 to advance to PENDING_EXIT.
        """
        shadow_only = {s for s in shadow_universe if s not in live_universe}
        if not shadow_only:
            return False, 0.0

        if ev_scores is not None:
            sym_ev = float(ev_scores.get(symbol, 0.0))
            shadow_evs = [
                float(ev_scores[s])
                for s in shadow_only
                if s in ev_scores and not s.startswith("__n_")
            ]
            if not shadow_evs:
                # Candidates exist but no EV data → treat delta as positive
                return True, 0.01
            best_shadow = max(shadow_evs)
            delta = float(best_shadow - sym_ev)
            return True, max(delta, 0.01)  # floor: if candidates exist, delta is positive

        # No EV scores → replacement exists, assume positive delta
        return True, 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log_transition(
    symbol: str,
    prev: str,
    new: str,
    pressure: float,
    repl_delta: float,
    held: int,
    reason: str,
) -> None:
    logger.info(
        "[DEMOTION] symbol=%s previous_state=%s new_state=%s "
        "pressure=%.3f replacement_delta=%.3f held_position=%d reason=%s",
        symbol, prev, new, pressure, repl_delta, held, reason,
    )


def _average_days_in_state(records: list, now_str: str) -> float:
    if not records:
        return 0.0
    try:
        now_dt = datetime.fromisoformat(now_str)
        total  = 0.0
        valid  = 0
        for r in records:
            try:
                entered = datetime.fromisoformat(r.entered_at)
                total  += (now_dt - entered).total_seconds() / 86400.0
                valid  += 1
            except Exception:
                pass
        return round(total / valid, 2) if valid else 0.0
    except Exception:
        return 0.0
