"""
src/universe/rotation_planner.py — Phase 3B: Rotation Planner

Produces deterministic rotation plans that maintain TARGET_UNIVERSE_SIZE.

Key properties:
  - TARGET_UNIVERSE_SIZE always maintained: fills deficit with best candidates
  - Staleness protection: candidates older than MAX_CANDIDATE_AGE_DAYS rejected
  - Plan lifecycle: PROPOSED → APPROVED → EXECUTED (or CANCELLED)
  - Outcome hook: rotation_plan_history.jsonl with outcome_status=PENDING
  - Determinism: plan_id = sha256(canonical decision inputs); same inputs → identical plan
  - Replay: decision_inputs field in rotation_plan.json sufficient to reconstruct decision

Separation:
  - Observation and planning only; no execution, no sell orders
  - FAIL_OPEN: write errors logged, not raised; caller receives plan regardless
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
TARGET_UNIVERSE_SIZE: int   = 42   # default; override via constructor
MAX_CANDIDATE_AGE_DAYS: int = 7    # reject candidates scored > this many days ago

# ── Plan lifecycle states ──────────────────────────────────────────────────────
PLAN_STATUS_PROPOSED  = "PROPOSED"
PLAN_STATUS_APPROVED  = "APPROVED"
PLAN_STATUS_EXECUTED  = "EXECUTED"
PLAN_STATUS_CANCELLED = "CANCELLED"

_VALID_STATUSES = frozenset({
    PLAN_STATUS_PROPOSED, PLAN_STATUS_APPROVED,
    PLAN_STATUS_EXECUTED, PLAN_STATUS_CANCELLED,
})

# ── Outcome hook status ────────────────────────────────────────────────────────
OUTCOME_STATUS_PENDING  = "PENDING"
OUTCOME_STATUS_REALIZED = "REALIZED"

# ── Scoring ────────────────────────────────────────────────────────────────────
_W_EV  = 0.60
_W_RSR = 0.40    # RSR raw score normalised by /100 before weighting

_SCHEMA_VERSION = 1


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CandidateRecord:
    symbol: str
    sector: str
    ev_score: float
    rsr_score: float          # raw [0, 100]
    composite_score: float    # _W_EV * ev_score + _W_RSR * (rsr_score / 100)
    scored_at: str            # ISO8601 when candidate was last scored
    age_days: float           # days since scored_at relative to reference_date
    is_stale: bool            # age_days > MAX_CANDIDATE_AGE_DAYS

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "CandidateRecord":
        return CandidateRecord(
            symbol=str(d["symbol"]),
            sector=str(d.get("sector", "")),
            ev_score=float(d.get("ev_score", 0.0)),
            rsr_score=float(d.get("rsr_score", 0.0)),
            composite_score=float(d.get("composite_score", 0.0)),
            scored_at=str(d.get("scored_at", "")),
            age_days=float(d.get("age_days", 0.0)),
            is_stale=bool(d.get("is_stale", False)),
        )


@dataclass
class RotationPlan:
    """Full rotation plan — rotation_plan.json."""
    plan_id: str                         # sha256(canonical decision inputs)
    created_at: str                      # ISO8601 UTC
    plan_status: str                     # PROPOSED / APPROVED / EXECUTED / CANCELLED
    target_universe_size: int
    current_live_size: int               # len(live_universe) before any removal
    post_removal_size: int               # current_live_size - len(symbols_to_remove)
    deficit: int                         # target - post_removal_size (slots to fill)
    symbols_to_remove: List[str]         # from demotion actuation
    symbols_to_add: List[str]            # selected replacements (len ≤ deficit)
    rotation_score: float                # mean composite of selected; 0.0 if none
    replacement_candidates: List[CandidateRecord]   # accepted, ranked by score
    rejected_candidates: List[CandidateRecord]      # stale or cutoff
    decision_inputs: dict                            # all inputs for replay
    schema_version: int = _SCHEMA_VERSION

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @staticmethod
    def from_dict(d: dict) -> "RotationPlan":
        return RotationPlan(
            plan_id=d["plan_id"],
            created_at=d["created_at"],
            plan_status=d["plan_status"],
            target_universe_size=int(d["target_universe_size"]),
            current_live_size=int(d["current_live_size"]),
            post_removal_size=int(d["post_removal_size"]),
            deficit=int(d["deficit"]),
            symbols_to_remove=list(d["symbols_to_remove"]),
            symbols_to_add=list(d["symbols_to_add"]),
            rotation_score=float(d["rotation_score"]),
            replacement_candidates=[
                CandidateRecord.from_dict(c)
                for c in d.get("replacement_candidates", [])
            ],
            rejected_candidates=[
                CandidateRecord.from_dict(c)
                for c in d.get("rejected_candidates", [])
            ],
            decision_inputs=dict(d.get("decision_inputs", {})),
            schema_version=int(d.get("schema_version", _SCHEMA_VERSION)),
        )


@dataclass
class RotationPlanHistoryRecord:
    """Outcome-hook record written to rotation_plan_history.jsonl."""
    plan_id: str
    created_at: str
    plan_status: str
    symbols_to_remove: List[str]
    symbols_to_add: List[str]
    rotation_score: float
    outcome_status: str          # PENDING (future outcome tracking fills this)
    target_universe_size: int
    post_rotation_size: int      # expected universe size after rotation executes
    schema_version: int = _SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Planner
# ─────────────────────────────────────────────────────────────────────────────

class RotationPlanner:
    """
    Builds deterministic rotation plans that maintain TARGET_UNIVERSE_SIZE.

    Usage:
        planner = RotationPlanner(plan_file, history_file)
        plan = planner.plan(live_universe, shadow_universe, symbols_to_remove, ...)
        planner.write_plan(plan)
        planner.write_history(plan)
    """

    def __init__(
        self,
        rotation_plan_file: Path,
        rotation_plan_history_file: Path,
        target_universe_size: int = TARGET_UNIVERSE_SIZE,
        max_candidate_age_days: int = MAX_CANDIDATE_AGE_DAYS,
    ) -> None:
        self._plan_file    = rotation_plan_file
        self._history_file = rotation_plan_history_file
        self._target       = target_universe_size
        self._max_age      = max_candidate_age_days

    # ── Public API ─────────────────────────────────────────────────────────────

    def plan(
        self,
        live_universe: Dict[str, str],
        shadow_universe: Dict[str, str],
        symbols_to_remove: Optional[List[str]] = None,
        ev_scores: Optional[Dict[str, float]] = None,
        rsr_scores: Optional[Dict[str, float]] = None,
        candidate_scored_at: Optional[Dict[str, str]] = None,
        reference_date: Optional[str] = None,
    ) -> RotationPlan:
        """
        Build a rotation plan.

        Args:
            live_universe:       current live symbols {symbol: sector}
            shadow_universe:     candidate pool {symbol: sector}
            symbols_to_remove:   symbols being demoted (may overlap live_universe)
            ev_scores:           EV composite scores per symbol
            rsr_scores:          RSR raw scores [0–100] per symbol
            candidate_scored_at: {symbol: ISO8601} — when each candidate was scored
            reference_date:      ISO8601 date string for staleness calc (default: today UTC)
        """
        removes = sorted(symbols_to_remove or [])
        now     = _now_utc()
        ref_dt  = _parse_date(reference_date) if reference_date else datetime.now(timezone.utc)
        ev      = {k: float(v) for k, v in (ev_scores or {}).items() if not k.startswith("__n_")}
        rsr     = {k: float(v) for k, v in (rsr_scores or {}).items()}
        scored  = dict(candidate_scored_at or {})

        # ── Universe sizing ────────────────────────────────────────────────────
        current_live_size = len(live_universe)
        post_removal_size = max(0, current_live_size - len(removes))
        deficit           = max(0, self._target - post_removal_size)

        # ── Candidate pool: shadow symbols not currently in live universe ──────
        shadow_only = {
            sym: sec
            for sym, sec in shadow_universe.items()
            if sym not in live_universe
        }

        # ── Score and classify candidates ─────────────────────────────────────
        accepted: List[CandidateRecord] = []
        rejected: List[CandidateRecord] = []

        for sym in sorted(shadow_only):
            sec        = shadow_only[sym]
            ev_s       = float(ev.get(sym, 0.0))
            rsr_s      = float(rsr.get(sym, 0.0))
            composite  = round(_W_EV * ev_s + _W_RSR * (rsr_s / 100.0), 6)
            scored_iso = scored.get(sym, "")
            age, stale = _staleness(scored_iso, ref_dt, self._max_age)

            rec = CandidateRecord(
                symbol=sym, sector=sec,
                ev_score=round(ev_s, 6),
                rsr_score=round(rsr_s, 4),
                composite_score=composite,
                scored_at=scored_iso,
                age_days=round(age, 2),
                is_stale=stale,
            )
            (rejected if stale else accepted).append(rec)

        # ── Rank accepted: composite desc, symbol asc (deterministic) ─────────
        accepted.sort(key=lambda c: (-c.composite_score, c.symbol))

        # ── Fill deficit — never exceed available accepted candidates ──────────
        to_add    = accepted[:deficit]
        symbols_to_add = [c.symbol for c in to_add]

        # ── rotation_score: mean composite of selected ─────────────────────────
        rotation_score = (
            round(sum(c.composite_score for c in to_add) / len(to_add), 6)
            if to_add else 0.0
        )

        # ── decision_inputs (for replay) ───────────────────────────────────────
        decision_inputs: dict = {
            "live_universe_keys":    sorted(live_universe.keys()),
            "shadow_universe_keys":  sorted(shadow_only.keys()),
            "symbols_to_remove":     removes,
            "target_universe_size":  self._target,
            "max_candidate_age_days": self._max_age,
            "reference_date":        ref_dt.date().isoformat(),
            "ev_scores":             {k: ev[k] for k in sorted(ev)},
            "rsr_scores":            {k: rsr[k] for k in sorted(rsr)},
            "candidate_scored_at":   {k: scored[k] for k in sorted(scored)},
            "scoring_weights":       {"ev": _W_EV, "rsr": _W_RSR},
            "schema_version":        _SCHEMA_VERSION,
        }
        plan_id = _sha256(_canonical(decision_inputs))

        return RotationPlan(
            plan_id=plan_id,
            created_at=now,
            plan_status=PLAN_STATUS_PROPOSED,
            target_universe_size=self._target,
            current_live_size=current_live_size,
            post_removal_size=post_removal_size,
            deficit=deficit,
            symbols_to_remove=removes,
            symbols_to_add=symbols_to_add,
            rotation_score=rotation_score,
            replacement_candidates=to_add,
            rejected_candidates=rejected,
            decision_inputs=decision_inputs,
        )

    def write_plan(self, plan: RotationPlan) -> None:
        """Atomic write of rotation_plan.json. FAIL_OPEN."""
        try:
            content = json.dumps(plan.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
            _write_atomic(self._plan_file, content)
            logger.info(
                "[ROTATION_PLANNER] plan written: plan_id=%s status=%s add=%s remove=%s",
                plan.plan_id[:12], plan.plan_status,
                plan.symbols_to_add, plan.symbols_to_remove,
            )
        except Exception as exc:
            logger.warning("[ROTATION_PLANNER] write_plan failed (FAIL_OPEN): %s", exc)

    def write_history(self, plan: RotationPlan) -> None:
        """Append outcome-hook record to rotation_plan_history.jsonl. FAIL_OPEN."""
        try:
            rec = RotationPlanHistoryRecord(
                plan_id=plan.plan_id,
                created_at=plan.created_at,
                plan_status=plan.plan_status,
                symbols_to_remove=plan.symbols_to_remove,
                symbols_to_add=plan.symbols_to_add,
                rotation_score=plan.rotation_score,
                outcome_status=OUTCOME_STATUS_PENDING,
                target_universe_size=plan.target_universe_size,
                post_rotation_size=plan.post_removal_size + len(plan.symbols_to_add),
            )
            _append_jsonl(self._history_file, rec.to_dict())
            logger.info(
                "[ROTATION_PLANNER] history appended: plan_id=%s outcome_status=%s",
                plan.plan_id[:12], OUTCOME_STATUS_PENDING,
            )
        except Exception as exc:
            logger.warning("[ROTATION_PLANNER] write_history failed (FAIL_OPEN): %s", exc)

    def load_plan(self) -> Optional[RotationPlan]:
        """Load current rotation_plan.json. Returns None if absent or corrupt."""
        if not self._plan_file.exists():
            return None
        try:
            return RotationPlan.from_dict(
                json.loads(self._plan_file.read_text(encoding="utf-8"))
            )
        except Exception as exc:
            logger.warning("[ROTATION_PLANNER] load_plan failed: %s", exc)
            return None

    def update_plan_status(self, plan_id: str, new_status: str) -> bool:
        """
        Load current plan, verify plan_id, update status, write atomically.
        FAIL_OPEN: returns False on error.

        Valid transitions:
          PROPOSED  → APPROVED | CANCELLED
          APPROVED  → EXECUTED | CANCELLED
          EXECUTED  → (terminal)
          CANCELLED → (terminal)
        """
        if new_status not in _VALID_STATUSES:
            logger.warning("[ROTATION_PLANNER] invalid status: %s", new_status)
            return False
        plan = self.load_plan()
        if plan is None:
            logger.warning("[ROTATION_PLANNER] no plan on disk to update")
            return False
        if plan.plan_id != plan_id:
            logger.warning(
                "[ROTATION_PLANNER] plan_id mismatch: on-disk=%s requested=%s",
                plan.plan_id[:12], plan_id[:12],
            )
            return False
        if plan.plan_status in (PLAN_STATUS_EXECUTED, PLAN_STATUS_CANCELLED):
            logger.warning(
                "[ROTATION_PLANNER] plan in terminal state %s; no update", plan.plan_status
            )
            return False
        plan.plan_status = new_status
        self.write_plan(plan)
        return True

    def replay_from_plan(self, plan: RotationPlan) -> RotationPlan:
        """
        Reconstruct rotation decision from decision_inputs stored in plan.
        Returns a new RotationPlan built from the stored inputs.
        Replayed plan has same plan_id and identical outputs (except created_at).
        """
        di = plan.decision_inputs
        live_keys   = di.get("live_universe_keys", [])
        shadow_keys = di.get("shadow_universe_keys", [])
        ev_scores   = di.get("ev_scores", {})
        rsr_scores  = di.get("rsr_scores", {})
        scored_at   = di.get("candidate_scored_at", {})
        removes     = di.get("symbols_to_remove", [])
        ref_date    = di.get("reference_date", "")

        # Reconstruct minimal universe dicts for replay
        live_uni   = {s: "" for s in live_keys}
        shadow_uni = {s: "" for s in shadow_keys}

        replayed_planner = RotationPlanner(
            rotation_plan_file=self._plan_file,
            rotation_plan_history_file=self._history_file,
            target_universe_size=di.get("target_universe_size", self._target),
            max_candidate_age_days=di.get("max_candidate_age_days", self._max_age),
        )
        return replayed_planner.plan(
            live_universe=live_uni,
            shadow_universe=shadow_uni,
            symbols_to_remove=removes,
            ev_scores=ev_scores,
            rsr_scores=rsr_scores,
            candidate_scored_at=scored_at,
            reference_date=ref_date,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(d: dict) -> str:
    return json.dumps(d, sort_keys=True, ensure_ascii=False)


def _sha256(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parse_date(s: str) -> datetime:
    """Parse ISO date or datetime string → timezone-aware datetime."""
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return datetime.now(timezone.utc)


def _staleness(
    scored_iso: str,
    ref_dt: datetime,
    max_age_days: int,
) -> tuple[float, bool]:
    """
    Returns (age_days, is_stale).
    If scored_iso is empty or unparseable → age=0.0, stale=False (no data → accept).
    """
    if not scored_iso:
        return 0.0, False
    try:
        scored_dt = _parse_date(scored_iso)
        age = (ref_dt - scored_dt).total_seconds() / 86400.0
        age = max(0.0, age)
        return round(age, 4), age > max_age_days
    except Exception:
        return 0.0, False


def _write_atomic(path: Path, content: str) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
