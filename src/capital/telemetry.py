"""
telemetry.py — Capital scaling structured telemetry

Schema-versioned, append-only JSON lines.
Replay-safe: pure write, no read-modify-write.

Schema v2 additions (Phase 2 adaptive growth):
  opportunity_cost_bps, idle_capital_ratio, edge_persistence_score,
  aggressiveness_score, effective_independent_positions,
  post_cost_alpha_retention, deployment_regime_state,
  multihorizon_aggregate_confidence
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

JST = timezone(timedelta(hours=9))
TELEMETRY_SCHEMA_VERSION = 2


@dataclass
class CapitalScalingTelemetryEntry:
    ts: str
    trading_day: str
    schema_version: int

    # Capital state
    actual_equity: float
    effective_capital: float
    deployable_capital: float
    risk_adjusted_capital: float
    capital_growth_rate: float

    # Utilization
    capital_utilization: float        # deployed_value / risk_adjusted_capital
    participation_rate: float         # avg across orders
    fill_ratio: float
    partial_fill_ratio: float
    slippage_bps: float
    reject_rate: float
    execution_latency_ms: float
    liquidity_stress_score: float

    # Shadow simulation
    shadow_capacity_metrics: dict     # {tier_label: ShadowTierResult-as-dict}

    # State flags
    deployment_ramp_state: str        # RAMPING | FROZEN | RECOVERING
    capital_freeze_state: str         # NORMAL | FROZEN
    capital_freeze_reason: Optional[str]

    # CAGR-oriented adaptive growth metrics (Phase 2)
    opportunity_cost_bps: float = 0.0          # alpha missed due to idle capital
    idle_capital_ratio: float = 0.0            # idle / effective [0,1]
    edge_persistence_score: float = 0.5        # composite_edge_score from edge_model
    aggressiveness_score: float = 0.5          # aggression_ema from AggressionState
    effective_independent_positions: float = 0.0  # Herfindahl effective_N
    post_cost_alpha_retention: float = 0.0     # net_alpha / gross_alpha [0,1]
    deployment_regime_state: str = "RAMPING"   # from DeploymentStateRecord
    multihorizon_aggregate_confidence: float = 0.63  # weighted geometric mean


def build_telemetry_entry(
    actual_equity: float,
    effective_capital: float,
    deployable_capital: float,
    risk_adjusted_capital: float,
    capital_growth_rate: float,
    deployed_value: float,
    participation_rate: float,
    fill_ratio: float,
    partial_fill_ratio: float,
    slippage_bps: float,
    reject_rate: float,
    execution_latency_ms: float,
    liquidity_stress_score: float,
    shadow_capacity_metrics: dict,
    deployment_ramp_state: str,
    capital_freeze_state: str,
    capital_freeze_reason: Optional[str] = None,
    trading_day: Optional[str] = None,
    # Phase 2 optional fields
    opportunity_cost_bps: float = 0.0,
    idle_capital_ratio: float = 0.0,
    edge_persistence_score: float = 0.5,
    aggressiveness_score: float = 0.5,
    effective_independent_positions: float = 0.0,
    post_cost_alpha_retention: float = 0.0,
    deployment_regime_state: str = "RAMPING",
    multihorizon_aggregate_confidence: float = 0.63,
) -> CapitalScalingTelemetryEntry:
    if trading_day is None:
        trading_day = datetime.now(JST).date().isoformat()

    utilization = (
        deployed_value / risk_adjusted_capital
        if risk_adjusted_capital > 0.0
        else 0.0
    )

    return CapitalScalingTelemetryEntry(
        ts=datetime.now(JST).isoformat(),
        trading_day=trading_day,
        schema_version=TELEMETRY_SCHEMA_VERSION,
        actual_equity=round(actual_equity, 0),
        effective_capital=round(effective_capital, 0),
        deployable_capital=round(deployable_capital, 0),
        risk_adjusted_capital=round(risk_adjusted_capital, 0),
        capital_growth_rate=round(capital_growth_rate, 6),
        capital_utilization=round(min(1.5, utilization), 4),  # allow >1.0 for diagnostics
        participation_rate=round(participation_rate, 4),
        fill_ratio=round(fill_ratio, 4),
        partial_fill_ratio=round(partial_fill_ratio, 4),
        slippage_bps=round(slippage_bps, 2),
        reject_rate=round(reject_rate, 4),
        execution_latency_ms=round(execution_latency_ms, 1),
        liquidity_stress_score=round(liquidity_stress_score, 4),
        shadow_capacity_metrics=shadow_capacity_metrics,
        deployment_ramp_state=deployment_ramp_state,
        capital_freeze_state=capital_freeze_state,
        capital_freeze_reason=capital_freeze_reason,
        opportunity_cost_bps=round(opportunity_cost_bps, 4),
        idle_capital_ratio=round(max(0.0, min(1.0, idle_capital_ratio)), 4),
        edge_persistence_score=round(max(0.0, min(1.0, edge_persistence_score)), 4),
        aggressiveness_score=round(max(0.0, min(2.0, aggressiveness_score)), 4),
        effective_independent_positions=round(max(0.0, effective_independent_positions), 4),
        post_cost_alpha_retention=round(max(0.0, min(1.0, post_cost_alpha_retention)), 4),
        deployment_regime_state=deployment_regime_state,
        multihorizon_aggregate_confidence=round(max(0.0, min(1.0, multihorizon_aggregate_confidence)), 4),
    )


def append_telemetry(entry: CapitalScalingTelemetryEntry, path: Path) -> None:
    """Append-only write. One JSON line per call."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
    logger.debug("capital telemetry appended: %s", path)
