"""
promotion_governance.py — Deterministic alpha promotion governance

Promotion eligibility requires ALL criteria to pass.
The runtime favors allocator survivability over aggressive promotion.

Promotion states:
  SHADOW → CANDIDATE → PRODUCTION

All decisions are immutable and append-only.
Replay must reproduce identical promotion outcomes.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
SCHEMA_VERSION: int = 1


# ── Promotion criteria ─────────────────────────────────────────────────────────

@dataclass
class PromotionCriteria:
    """Configurable thresholds. Defaults are conservative."""
    min_observation_days: int = 20         # shadow must run ≥ N days
    min_completed_trades: int = 5          # ≥ N closed trades
    min_post_cost_sharpe: float = 0.50     # Sharpe in shadow ≥ threshold
    max_drawdown_fraction: float = 0.20    # max DD ≤ threshold
    min_stability_score: float = 0.60      # win-rate consistency ≥ threshold
    min_post_cost_retention: float = 0.50  # ≥ 50% of gross alpha survives costs
    schema_version: int = SCHEMA_VERSION


# ── Promotion decision ─────────────────────────────────────────────────────────

@dataclass
class PromotionDecision:
    """Immutable promotion decision. Append-only."""
    candidate_id: str
    timestamp: str
    decision: str          # ELIGIBLE | INELIGIBLE | PROMOTED | DEFERRED
    current_status: str
    target_status: str
    # Per-criterion results
    observation_days_ok: bool
    trades_ok: bool
    sharpe_ok: bool
    drawdown_ok: bool
    stability_ok: bool
    retention_ok: bool
    all_criteria_met: bool
    reason: str
    schema_version: int = SCHEMA_VERSION


# ── Evaluation ─────────────────────────────────────────────────────────────────

@dataclass
class ShadowPerformanceSummary:
    """Shadow metrics consumed by the promotion evaluator."""
    candidate_id: str
    observation_days: int
    completed_trades: int
    post_cost_sharpe: float
    max_drawdown: float
    stability_score: float       # win-rate consistency [0, 1]
    post_cost_retention: float   # fraction of gross alpha retained [0, 1]


def evaluate_promotion(
    summary: ShadowPerformanceSummary,
    criteria: PromotionCriteria,
    timestamp: str,
    current_status: str,
    target_status: str,
) -> PromotionDecision:
    """
    Deterministic promotion evaluation.
    All criteria must pass for ELIGIBLE outcome.
    Same inputs → same decision on any replay.
    """
    obs_ok = summary.observation_days >= criteria.min_observation_days
    trades_ok = summary.completed_trades >= criteria.min_completed_trades
    sharpe_ok = summary.post_cost_sharpe >= criteria.min_post_cost_sharpe
    dd_ok = summary.max_drawdown <= criteria.max_drawdown_fraction
    stab_ok = summary.stability_score >= criteria.min_stability_score
    ret_ok = summary.post_cost_retention >= criteria.min_post_cost_retention
    all_met = all([obs_ok, trades_ok, sharpe_ok, dd_ok, stab_ok, ret_ok])

    failed = []
    if not obs_ok:
        failed.append(f"observation_days={summary.observation_days}<{criteria.min_observation_days}")
    if not trades_ok:
        failed.append(f"trades={summary.completed_trades}<{criteria.min_completed_trades}")
    if not sharpe_ok:
        failed.append(f"sharpe={summary.post_cost_sharpe:.2f}<{criteria.min_post_cost_sharpe}")
    if not dd_ok:
        failed.append(f"maxdd={summary.max_drawdown:.2f}>{criteria.max_drawdown_fraction}")
    if not stab_ok:
        failed.append(f"stability={summary.stability_score:.2f}<{criteria.min_stability_score}")
    if not ret_ok:
        failed.append(f"retention={summary.post_cost_retention:.2f}<{criteria.min_post_cost_retention}")

    if all_met:
        decision = "ELIGIBLE"
        reason = "all criteria met"
    else:
        decision = "INELIGIBLE"
        reason = "; ".join(failed)

    return PromotionDecision(
        candidate_id=summary.candidate_id,
        timestamp=timestamp,
        decision=decision,
        current_status=current_status,
        target_status=target_status,
        observation_days_ok=obs_ok,
        trades_ok=trades_ok,
        sharpe_ok=sharpe_ok,
        drawdown_ok=dd_ok,
        stability_ok=stab_ok,
        retention_ok=ret_ok,
        all_criteria_met=all_met,
        reason=reason,
    )


def compute_shadow_sharpe(observations: list, risk_free_bps: float = 0.0) -> float:
    """
    Compute Sharpe ratio from shadow observations.
    Uses edge_proxy_bps as the return series.
    Returns 0.0 if insufficient data.
    """
    returns = [o.edge_proxy_bps - risk_free_bps for o in observations]
    n = len(returns)
    if n < 2:
        return 0.0
    mean = sum(returns) / n
    variance = sum((r - mean) ** 2 for r in returns) / (n - 1)
    std = variance ** 0.5
    if std <= 0.0:
        return 0.0
    return round(mean / std, 4)


def compute_stability_from_observations(observations: list) -> float:
    """
    Stability = consistency of positive edge across observations.
    Uses post_cost_retention as proxy.
    """
    if not observations:
        return 0.0
    positive = sum(1 for o in observations if o.edge_proxy_bps > 0)
    win_rate = positive / len(observations)
    # Weight by consistency: penalize high variance in retention
    retentions = [o.post_cost_retention for o in observations]
    if len(retentions) < 2:
        return win_rate
    mean_r = sum(retentions) / len(retentions)
    std_r = (sum((r - mean_r) ** 2 for r in retentions) / (len(retentions) - 1)) ** 0.5
    stability = win_rate * (1.0 - min(1.0, std_r * 2))
    return round(max(0.0, stability), 4)


# ── Persistence ────────────────────────────────────────────────────────────────

def append_promotion_decision(decision: PromotionDecision, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(asdict(decision), ensure_ascii=False, sort_keys=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_promotion_decisions(path: Path) -> list[PromotionDecision]:
    if not path.exists():
        return []
    decisions: list[PromotionDecision] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            decisions.append(PromotionDecision(
                candidate_id=data.get("candidate_id", ""),
                timestamp=data.get("timestamp", ""),
                decision=data.get("decision", ""),
                current_status=data.get("current_status", ""),
                target_status=data.get("target_status", ""),
                observation_days_ok=bool(data.get("observation_days_ok", False)),
                trades_ok=bool(data.get("trades_ok", False)),
                sharpe_ok=bool(data.get("sharpe_ok", False)),
                drawdown_ok=bool(data.get("drawdown_ok", False)),
                stability_ok=bool(data.get("stability_ok", False)),
                retention_ok=bool(data.get("retention_ok", False)),
                all_criteria_met=bool(data.get("all_criteria_met", False)),
                reason=data.get("reason", ""),
            ))
        except Exception as exc:
            logger.warning("promotion_decision parse error (%s) — skipped", exc)
    return decisions
