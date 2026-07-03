"""
explainability.py — Allocation decision rationale layer

Persists immutable, deterministic allocation explanations.
Supports replay analysis, allocator debugging, drift detection,
and deployment explainability.

All records are append-only JSONL. Never modified after write.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SCHEMA_VERSION: int = 1


@dataclass
class AllocationDecision:
    """Immutable snapshot of one allocation decision."""
    timestamp: str                       # ISO-8601
    trading_day: str                     # YYYY-MM-DD
    symbol: str
    decision: str                        # "SELECTED" | "CANDIDATE" | "REJECTED"
    rank: int                            # efficiency rank (1 = best)

    # Scoring factors
    capital_efficiency_score: float
    expected_geometric_return: float
    liquidity_penalty: float
    correlation_penalty: float
    drawdown_penalty: float
    correlation_with_portfolio: float

    # Context at time of decision
    edge_persistence: float              # from alpha_lifecycle [0, 1]
    regime_confidence: float             # from multihorizon [0, 1]
    post_cost_alpha_bps: float
    aggressiveness_state: str            # AGGRESSIVE|DEFENSIVE|OPPORTUNISTIC|EXPLORATION
    concentration_state: str             # LOW|MEDIUM|HIGH
    effective_independent_n: float       # from exposure_core

    rejection_reason: str = ""           # non-empty only when REJECTED
    schema_version: int = SCHEMA_VERSION


def record_to_dict(decision: AllocationDecision) -> dict:
    """Deterministic dict serialization."""
    return asdict(decision)


def append_allocation_decision(decision: AllocationDecision, path: Path) -> None:
    """Append one decision record to append-only JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record_to_dict(decision), ensure_ascii=False, sort_keys=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_allocation_decisions(path: Path) -> list[AllocationDecision]:
    """Replay all allocation decisions from JSONL. Returns empty list if missing."""
    if not path.exists():
        return []
    decisions: list[AllocationDecision] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            decisions.append(_decision_from_dict(data))
    except Exception as exc:
        logger.warning("allocation_decisions load failed (%s)", exc)
    return decisions


def load_recent_decisions(path: Path, n: int = 100) -> list[AllocationDecision]:
    """Load last N decisions efficiently without loading entire file."""
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        recent_lines = [l for l in lines[-n:] if l.strip()]
        return [_decision_from_dict(json.loads(l)) for l in recent_lines]
    except Exception as exc:
        logger.warning("recent decisions load failed (%s)", exc)
        return []


def _decision_from_dict(data: dict) -> AllocationDecision:
    return AllocationDecision(
        timestamp=data.get("timestamp", ""),
        trading_day=data.get("trading_day", ""),
        symbol=data.get("symbol", ""),
        decision=data.get("decision", "UNKNOWN"),
        rank=int(data.get("rank", 0)),
        capital_efficiency_score=float(data.get("capital_efficiency_score", 0.0)),
        expected_geometric_return=float(data.get("expected_geometric_return", 0.0)),
        liquidity_penalty=float(data.get("liquidity_penalty", 1.0)),
        correlation_penalty=float(data.get("correlation_penalty", 1.0)),
        drawdown_penalty=float(data.get("drawdown_penalty", 1.0)),
        correlation_with_portfolio=float(data.get("correlation_with_portfolio", 0.0)),
        edge_persistence=float(data.get("edge_persistence", 0.5)),
        regime_confidence=float(data.get("regime_confidence", 0.5)),
        post_cost_alpha_bps=float(data.get("post_cost_alpha_bps", 0.0)),
        aggressiveness_state=data.get("aggressiveness_state", "UNKNOWN"),
        concentration_state=data.get("concentration_state", "UNKNOWN"),
        effective_independent_n=float(data.get("effective_independent_n", 0.0)),
        rejection_reason=data.get("rejection_reason", ""),
        schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
    )


def build_allocation_decision(
    *,
    timestamp: str,
    trading_day: str,
    symbol: str,
    decision: str,
    rank: int,
    efficiency_score,       # EfficiencyScore from efficiency_ranker
    edge_persistence: float,
    regime_confidence: float,
    post_cost_alpha_bps: float,
    aggressiveness_state: str,
    concentration_state: str,
    effective_independent_n: float,
    rejection_reason: str = "",
) -> AllocationDecision:
    """
    Convenience constructor — takes EfficiencyScore directly.
    All numeric fields bounded to avoid serialization surprises.
    """
    return AllocationDecision(
        timestamp=timestamp,
        trading_day=trading_day,
        symbol=symbol,
        decision=decision,
        rank=rank,
        capital_efficiency_score=round(float(efficiency_score.capital_efficiency_score), 6),
        expected_geometric_return=round(float(efficiency_score.expected_geometric_return), 6),
        liquidity_penalty=round(float(efficiency_score.liquidity_penalty), 4),
        correlation_penalty=round(float(efficiency_score.correlation_penalty), 4),
        drawdown_penalty=round(float(efficiency_score.drawdown_penalty), 4),
        correlation_with_portfolio=round(float(efficiency_score.correlation_with_portfolio), 4),
        edge_persistence=round(max(0.0, min(1.0, float(edge_persistence))), 4),
        regime_confidence=round(max(0.0, min(1.0, float(regime_confidence))), 4),
        post_cost_alpha_bps=round(float(post_cost_alpha_bps), 2),
        aggressiveness_state=aggressiveness_state,
        concentration_state=concentration_state,
        effective_independent_n=round(max(0.0, float(effective_independent_n)), 4),
        rejection_reason=rejection_reason,
    )
