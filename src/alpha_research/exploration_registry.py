"""
exploration_registry.py — Append-only exploration candidate registry

Every alpha candidate discovered during research is registered here.
Historical records remain immutable even after retirement.

Lifecycle:
  DISCOVERED → SHADOW → CANDIDATE → PRODUCTION → RETIRED

Historical replayability is preserved for all lifecycle stages.
Corruption-tolerant: bad lines are skipped, not fatal.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
VALID_STATUSES = frozenset({
    "DISCOVERED", "SHADOW", "CANDIDATE", "PRODUCTION", "RETIRED"
})
VALID_TRANSITIONS: dict[str, frozenset] = {
    "DISCOVERED": frozenset({"SHADOW", "RETIRED"}),
    "SHADOW":     frozenset({"CANDIDATE", "RETIRED"}),
    "CANDIDATE":  frozenset({"PRODUCTION", "RETIRED"}),
    "PRODUCTION": frozenset({"RETIRED"}),
    "RETIRED":    frozenset(),   # terminal
}
SCHEMA_VERSION: int = 1


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class ExplorationCandidate:
    """Immutable-after-creation candidate descriptor."""
    candidate_id: str
    discovery_time: str               # ISO-8601
    source_strategy: str
    factor_count: int
    expected_holding_period: int      # days
    turnover_estimate: float          # annual fraction [0, 1]
    post_cost_expectancy_bps: float
    exploration_status: str           # DISCOVERED | SHADOW | CANDIDATE | PRODUCTION | RETIRED
    shadow_runtime_state: str         # INACTIVE | RUNNING | PAUSED
    promotion_attempts: int = 0
    retirement_reason: str = ""
    lifecycle_stage: str = "DISCOVERED"
    created_at: str = ""
    schema_version: int = SCHEMA_VERSION


@dataclass
class StatusChange:
    """Append-only status transition record."""
    candidate_id: str
    timestamp: str
    from_status: str
    to_status: str
    reason: str
    schema_version: int = SCHEMA_VERSION


# ── Validation ─────────────────────────────────────────────────────────────────

class InvalidTransitionError(ValueError):
    pass


def can_transition(from_status: str, to_status: str) -> bool:
    return to_status in VALID_TRANSITIONS.get(from_status, frozenset())


def validate_transition(candidate_id: str, from_status: str, to_status: str) -> None:
    if not can_transition(from_status, to_status):
        raise InvalidTransitionError(
            f"[{candidate_id}] invalid transition: {from_status} → {to_status}. "
            f"Allowed: {sorted(VALID_TRANSITIONS.get(from_status, set()))}"
        )


# ── Registry operations ────────────────────────────────────────────────────────

def register_candidate(candidate: ExplorationCandidate, path: Path) -> None:
    """Append candidate registration to the registry JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(asdict(candidate), ensure_ascii=False, sort_keys=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def record_status_change(change: StatusChange, path: Path) -> None:
    """Append an immutable status-change record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(asdict(change), ensure_ascii=False, sort_keys=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def transition_candidate(
    candidate: ExplorationCandidate,
    to_status: str,
    timestamp: str,
    reason: str,
    registry_path: Path,
) -> ExplorationCandidate:
    """
    Validate and apply a status transition.
    Returns updated candidate. Records change to registry.
    Raises InvalidTransitionError on illegal transition.
    """
    validate_transition(candidate.candidate_id, candidate.exploration_status, to_status)

    change = StatusChange(
        candidate_id=candidate.candidate_id,
        timestamp=timestamp,
        from_status=candidate.exploration_status,
        to_status=to_status,
        reason=reason,
    )
    record_status_change(change, registry_path)

    promotion_attempts = candidate.promotion_attempts
    if to_status == "PRODUCTION":
        promotion_attempts += 1

    return ExplorationCandidate(
        candidate_id=candidate.candidate_id,
        discovery_time=candidate.discovery_time,
        source_strategy=candidate.source_strategy,
        factor_count=candidate.factor_count,
        expected_holding_period=candidate.expected_holding_period,
        turnover_estimate=candidate.turnover_estimate,
        post_cost_expectancy_bps=candidate.post_cost_expectancy_bps,
        exploration_status=to_status,
        shadow_runtime_state=candidate.shadow_runtime_state,
        promotion_attempts=promotion_attempts,
        retirement_reason=reason if to_status == "RETIRED" else candidate.retirement_reason,
        lifecycle_stage=to_status,
        created_at=candidate.created_at,
    )


# ── Persistence ────────────────────────────────────────────────────────────────

def load_registry(path: Path) -> list[ExplorationCandidate]:
    """
    Replay full registry from JSONL.
    Corruption-tolerant: bad lines are logged and skipped.
    """
    if not path.exists():
        return []
    candidates: list[ExplorationCandidate] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            if "candidate_id" not in data:
                continue  # status-change record, not a candidate
            if data.get("exploration_status") not in VALID_STATUSES:
                logger.warning("registry line %d: invalid status — skipped", line_no)
                continue
            candidates.append(_candidate_from_dict(data))
        except Exception as exc:
            logger.warning("registry line %d: parse error (%s) — skipped", line_no, exc)
    return candidates


def load_status_changes(path: Path) -> list[StatusChange]:
    """Replay all status-change records from JSONL."""
    if not path.exists():
        return []
    changes: list[StatusChange] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            if "from_status" not in data:
                continue  # candidate registration record
            changes.append(StatusChange(
                candidate_id=data.get("candidate_id", ""),
                timestamp=data.get("timestamp", ""),
                from_status=data.get("from_status", ""),
                to_status=data.get("to_status", ""),
                reason=data.get("reason", ""),
            ))
        except Exception as exc:
            logger.warning("status_changes parse error (%s) — skipped", exc)
    return changes


def save_registry_snapshot(candidates: list[ExplorationCandidate], path: Path) -> None:
    """Save current candidate states as a point-in-time snapshot JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "candidates": [asdict(c) for c in candidates],
        "count": len(candidates),
        "by_status": {
            status: sum(1 for c in candidates if c.exploration_status == status)
            for status in VALID_STATUSES
        },
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _candidate_from_dict(data: dict) -> ExplorationCandidate:
    return ExplorationCandidate(
        candidate_id=data.get("candidate_id", ""),
        discovery_time=data.get("discovery_time", ""),
        source_strategy=data.get("source_strategy", ""),
        factor_count=int(data.get("factor_count", 0)),
        expected_holding_period=int(data.get("expected_holding_period", 5)),
        turnover_estimate=float(data.get("turnover_estimate", 0.0)),
        post_cost_expectancy_bps=float(data.get("post_cost_expectancy_bps", 0.0)),
        exploration_status=data.get("exploration_status", "DISCOVERED"),
        shadow_runtime_state=data.get("shadow_runtime_state", "INACTIVE"),
        promotion_attempts=int(data.get("promotion_attempts", 0)),
        retirement_reason=data.get("retirement_reason", ""),
        lifecycle_stage=data.get("lifecycle_stage", "DISCOVERED"),
        created_at=data.get("created_at", ""),
    )


def get_active_candidates(candidates: list[ExplorationCandidate]) -> list[ExplorationCandidate]:
    return [c for c in candidates if c.exploration_status not in ("RETIRED",)]


def get_by_status(candidates: list[ExplorationCandidate], status: str) -> list[ExplorationCandidate]:
    return [c for c in candidates if c.exploration_status == status]
