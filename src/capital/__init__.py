"""
capital/ — Capital-aware adaptive deployment governance layer

Phase 1: 4-layer capital hierarchy with rate-limited growth, fail-closed freeze,
         liquidity-constrained sizing, and shadow simulation.

Phase 2: Adaptive geometric-growth optimizer — aggression EMA, edge persistence,
         multi-horizon confidence, concentration penalty, alpha-after-impact gate,
         exploration budget, opportunity cost tracking, and reflexivity protection.
"""
from src.capital.capital_state import (
    CapitalState,
    compute_effective_capital,
    compute_deployable,
    compute_risk_adjusted,
    update_capital_state,
    load_capital_state,
    save_capital_state,
    DAILY_GROWTH_LIMIT_DEFAULT,
)
from src.capital.deployment_ramp import (
    DeploymentRampState,
    ExecutionQualityRecord,
    compute_quality_score,
    update_deployment_ramp,
    load_deployment_ramp,
    save_deployment_ramp,
)
from src.capital.capital_freeze import (
    FreezeState,
    FreezeConfig,
    evaluate_freeze,
    update_freeze_state,
    load_freeze_state,
    save_freeze_state,
)
from src.capital.market_impact import (
    MarketImpactConfig,
    estimate_market_impact_bps,
    impact_adjusted_price,
)
from src.capital.intraday_liquidity import (
    get_interval_liquidity_fraction,
    compute_interval_volume_yen,
    max_interval_order_value,
    compute_liquidity_stress_score,
)
from src.capital.risk_scalars import (
    ScalarConfig,
    compute_volatility_scalar,
    compute_liquidity_scalar,
    compute_execution_scalar,
)
from src.capital.position_sizer import (
    SizerConfig,
    SizedPosition,
    size_position,
)
from src.capital.admission_control import (
    AdmissionConfig,
    AdmissionDecision,
    assess_admission,
    filter_universe_by_capital,
)
from src.capital.shadow_simulation import (
    ShadowTierResult,
    ShadowSimulationRecord,
    simulate_tier,
    run_shadow_simulation,
    append_shadow_record,
)
from src.capital.telemetry import (
    CapitalScalingTelemetryEntry,
    build_telemetry_entry,
    append_telemetry,
)

# ── Phase 2: Adaptive growth modules ─────────────────────────────────────────
from src.capital.aggression import (
    AggressionInputs,
    AggressionState,
    compute_raw_aggression,
    update_aggression,
    compute_drawdown_penalty,
    compute_regime_score,
    load_aggression_state,
    save_aggression_state,
)
from src.capital.edge_model import (
    EdgeObservation,
    EdgeModelState,
    update_edge_model,
    build_edge_observation,
    load_edge_model,
    save_edge_model,
)
from src.capital.deployment_states import (
    DeploymentStateRecord,
    transition_deployment_state,
    get_growth_multiplier,
    load_deployment_state_record,
    save_deployment_state_record,
    STATE_GROWTH_MULTIPLIERS,
)
from src.capital.multihorizon import (
    MultiHorizonConfidence,
    update_multihorizon,
    load_multihorizon,
    save_multihorizon,
)
from src.capital.concentration import (
    ConcentrationMetrics,
    compute_effective_n,
    compute_concentration_penalty,
    compute_concentration_metrics,
    load_concentration_metrics,
    save_concentration_metrics,
)
from src.capital.alpha_impact import (
    AlphaImpactEstimate,
    estimate_net_alpha,
    estimate_gross_alpha_from_signal,
    compute_post_cost_alpha_retention,
)
from src.capital.exploration import (
    ExplorationState,
    ExplorationAllocation,
    compute_exploration_budget,
    update_exploration_state,
    assess_exploration_allocation,
    record_exploration_pnl,
    load_exploration_state,
    save_exploration_state,
)
from src.capital.opportunity_cost import (
    OpportunityCostRecord,
    OpportunityCostState,
    compute_opportunity_cost_bps,
    build_opportunity_cost_record,
    update_opportunity_cost_state,
    append_opportunity_cost_record,
    load_opportunity_cost_state,
)
from src.capital.reflexivity import (
    ReflexivityObservation,
    ReflexivityState,
    update_reflexivity,
    get_participation_rate,
    append_reflexivity_observation,
    load_reflexivity_state,
    save_reflexivity_state,
)

__all__ = [
    "CapitalState",
    "compute_effective_capital",
    "compute_deployable",
    "compute_risk_adjusted",
    "update_capital_state",
    "load_capital_state",
    "save_capital_state",
    "DAILY_GROWTH_LIMIT_DEFAULT",
    "DeploymentRampState",
    "ExecutionQualityRecord",
    "compute_quality_score",
    "update_deployment_ramp",
    "load_deployment_ramp",
    "save_deployment_ramp",
    "FreezeState",
    "FreezeConfig",
    "evaluate_freeze",
    "update_freeze_state",
    "load_freeze_state",
    "save_freeze_state",
    "MarketImpactConfig",
    "estimate_market_impact_bps",
    "impact_adjusted_price",
    "get_interval_liquidity_fraction",
    "compute_interval_volume_yen",
    "max_interval_order_value",
    "compute_liquidity_stress_score",
    "ScalarConfig",
    "compute_volatility_scalar",
    "compute_liquidity_scalar",
    "compute_execution_scalar",
    "SizerConfig",
    "SizedPosition",
    "size_position",
    "AdmissionConfig",
    "AdmissionDecision",
    "assess_admission",
    "filter_universe_by_capital",
    "ShadowTierResult",
    "ShadowSimulationRecord",
    "simulate_tier",
    "run_shadow_simulation",
    "append_shadow_record",
    "CapitalScalingTelemetryEntry",
    "build_telemetry_entry",
    "append_telemetry",
    # Phase 2
    "AggressionInputs",
    "AggressionState",
    "compute_raw_aggression",
    "update_aggression",
    "compute_drawdown_penalty",
    "compute_regime_score",
    "load_aggression_state",
    "save_aggression_state",
    "EdgeObservation",
    "EdgeModelState",
    "update_edge_model",
    "build_edge_observation",
    "load_edge_model",
    "save_edge_model",
    "DeploymentStateRecord",
    "transition_deployment_state",
    "get_growth_multiplier",
    "load_deployment_state_record",
    "save_deployment_state_record",
    "STATE_GROWTH_MULTIPLIERS",
    "MultiHorizonConfidence",
    "update_multihorizon",
    "load_multihorizon",
    "save_multihorizon",
    "ConcentrationMetrics",
    "compute_effective_n",
    "compute_concentration_penalty",
    "compute_concentration_metrics",
    "load_concentration_metrics",
    "save_concentration_metrics",
    "AlphaImpactEstimate",
    "estimate_net_alpha",
    "estimate_gross_alpha_from_signal",
    "compute_post_cost_alpha_retention",
    "ExplorationState",
    "ExplorationAllocation",
    "compute_exploration_budget",
    "update_exploration_state",
    "assess_exploration_allocation",
    "record_exploration_pnl",
    "load_exploration_state",
    "save_exploration_state",
    "OpportunityCostRecord",
    "OpportunityCostState",
    "compute_opportunity_cost_bps",
    "build_opportunity_cost_record",
    "update_opportunity_cost_state",
    "append_opportunity_cost_record",
    "load_opportunity_cost_state",
    "ReflexivityObservation",
    "ReflexivityState",
    "update_reflexivity",
    "get_participation_rate",
    "append_reflexivity_observation",
    "load_reflexivity_state",
    "save_reflexivity_state",
]
