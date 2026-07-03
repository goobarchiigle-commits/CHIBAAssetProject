"""
src/analytics/exit_intelligence/models.py — Exit Intelligence OS dataclasses

Immutable, deterministic, replay-compatible records.
Append-only analytics. No execution mutation.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

SCHEMA_VERSION: int = 1


def _sha256(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_json(d: dict) -> str:
    return json.dumps(d, sort_keys=True, ensure_ascii=False)


# ── Observation Layer ─────────────────────────────────────────────────────────

@dataclass
class ExitObservationRecord:
    """Final summarized exit record — written once at position close."""
    obs_id: str                        # sha256(symbol|entry_date|exit_date)
    symbol: str
    entry_date: str                    # YYYY-MM-DD JST
    exit_date: str                     # YYYY-MM-DD JST
    holding_days: int
    entry_price: float
    exit_price: float
    realized_return: float             # fraction
    realized_pnl_jpy: float
    exit_reason: str
    # final-day deterioration state
    unrealized_pnl_at_exit: float
    distance_from_peak_pct: float      # (peak_price - exit_price) / peak_price
    rs_rank_at_exit: float             # 0–100
    atr_pct_at_exit: float
    breadth_score_at_exit: float       # 0–1
    regime_at_exit: str
    exit_velocity_score: float         # ExitVelocityScore.total_score at exit
    # metadata
    path_snapshot_count: int           # number of ExitPathSnapshots recorded
    generated_at: str = ""
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def make_obs_id(cls, symbol: str, entry_date: str, exit_date: str) -> str:
        return _sha256(f"{symbol}|{entry_date}|{exit_date}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExitObservationRecord":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class ExitPathSnapshot:
    """Daily longitudinal deterioration snapshot for an open position."""
    snapshot_id: str                   # sha256(symbol|entry_date|snapshot_date)
    symbol: str
    entry_date: str
    snapshot_date: str                 # YYYY-MM-DD JST
    day_number: int                    # 1-based days since entry
    # price / PnL state
    close_price: float
    entry_price: float
    unrealized_pnl_fraction: float     # (close - entry) / entry
    unrealized_pnl_jpy: float
    peak_price_so_far: float
    distance_from_peak_pct: float      # (peak - close) / peak
    # deterioration signals
    rs_rank: float                     # 0–100
    atr_pct: float                     # ATR / close
    breadth_score: float               # 0–1 advancing ratio
    regime: str
    trend_persistence: float           # 0–1 slope consistency
    momentum: float                    # (close - close_20d) / close_20d
    momentum_prev: float               # prior day momentum
    volume_ratio: float                # today_vol / avg_vol_20d
    volatility_compression: float      # 0–1, 1=compressed
    upper_shadow_ratio: float          # upper_shadow / (high - low)
    exit_signal_count: int             # number of exit signals triggered today
    deterioration_velocity: float      # Δunrealized_pnl / Δday (fraction/day)
    generated_at: str = ""
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def make_snapshot_id(cls, symbol: str, entry_date: str, snapshot_date: str) -> str:
        return _sha256(f"{symbol}|{entry_date}|{snapshot_date}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExitPathSnapshot":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class ExecutionExitObservation:
    """Execution quality deterioration observation at exit fill."""
    exec_obs_id: str                   # sha256(symbol|exit_date|order_id)
    symbol: str
    exit_date: str
    order_id: str
    expected_exit_price: float
    actual_fill_price: float
    slippage_bps: float                # (actual - expected) / expected × 10000
    spread_bps: float
    liquidity_score: float             # 0–1 estimated fill quality
    is_auction_exit: bool
    is_gap_exit: bool                  # gapped past stop
    intraday_rejection_count: int      # attempts before fill
    time_to_fill_seconds: float
    volume_at_exit: float              # shares traded at exit bar
    generated_at: str = ""
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def make_exec_obs_id(cls, symbol: str, exit_date: str, order_id: str) -> str:
        return _sha256(f"{symbol}|{exit_date}|{order_id}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExecutionExitObservation":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class PortfolioExitPressureSnapshot:
    """Portfolio-level exit pressure snapshot (captured once per daily run)."""
    pressure_id: str                   # sha256(snapshot_date)
    snapshot_date: str
    concurrent_positions: int
    gross_exposure_jpy: float
    sector_concentration_max: float    # max sector weight 0–1
    signal_queue_pressure: float       # 0–1 new signals waiting
    replacement_pressure: float        # 0–1 better candidates queued
    portfolio_heat: float              # avg unrealized drawdown from peak across all positions
    cash_pressure: float               # 0=plenty of cash, 1=no cash
    capital_utilization: float         # deployed / total 0–1
    avg_holding_days: float
    worst_position_deterioration: float  # most negative distance_from_peak
    generated_at: str = ""
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def make_pressure_id(cls, snapshot_date: str) -> str:
        return _sha256(f"portfolio_pressure|{snapshot_date}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PortfolioExitPressureSnapshot":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


# ── Deterioration Modeling Layer ───────────────────────────────────────────────

@dataclass
class MomentumDecayMetrics:
    """Output of MomentumDecayAnalyzer."""
    symbol: str
    as_of_date: str
    momentum_velocity: float           # Δmomentum/day (negative = decaying)
    volume_deterioration: float        # 0–1 (0=fine, 1=severe)
    range_contraction: float           # 0–1 (1=contracting badly)
    upper_shadow_expansion: float      # 0–1
    breakout_exhaustion_score: float   # 0–1
    failed_continuation_count: int     # days of failed continuation in window

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StructuralFailureMetrics:
    """Output of StructuralFailureDetector."""
    symbol: str
    as_of_date: str
    breakout_failure: bool
    vwap_reclaim_failure: bool
    rs_collapse: bool                  # RS dropped > 10 ranks in 3 days
    breadth_deterioration: bool
    failed_trend_continuation: bool
    support_rejection: bool
    convexity_collapse: bool
    structural_failure_score: float    # 0–1 composite (more = worse)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PeakDecayVelocityMetrics:
    """Output of PeakDecayVelocityAnalyzer."""
    symbol: str
    as_of_date: str
    peak_decay_velocity: float         # (peak - current) / peak / days_from_peak
    drawdown_acceleration: float       # Δdecay_velocity/day
    days_from_peak: int
    deterioration_slope: float         # linear slope of distance_from_peak over window
    decay_classification: str          # "slow" | "orderly" | "fast_convex"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConvexityRetentionMetrics:
    """Output of ConvexityRetentionAnalyzer."""
    symbol: str
    as_of_date: str
    is_constructive_compression: bool
    is_high_tight_flag: bool
    is_persistent_leader: bool
    trend_continuation_quality: float  # 0–1
    convexity_retention_score: float   # 0–1 (1=fully retained)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PersistenceMetrics:
    """Output of PersistenceExtensionLayer."""
    symbol: str
    as_of_date: str
    trend_persistence_durability: float  # 0–1
    leader_persistence_score: float      # 0–1
    regime_persistence_score: float      # 0–1
    sector_leadership_continuity: float  # 0–1
    combined_persistence_score: float    # 0–1 weighted average

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Exit Pressure Layer ────────────────────────────────────────────────────────

@dataclass
class ExitVelocityScore:
    """Deterministic exit pressure score with explainable components."""
    score_id: str                      # sha256(symbol|scored_date)
    symbol: str
    scored_date: str
    # component scores (each 0–1 before weight)
    rs_deterioration: float
    breadth_decay: float
    breakout_failure: float
    vwap_rejection: float
    upper_shadow: float
    volume_decay: float
    volatility_compression: float      # inverted: 0=bad, 1=good → used as (1-val)
    momentum_decay: float
    distance_from_peak: float
    execution_deterioration: float
    portfolio_replacement_pressure: float
    # composite
    total_score: float                 # 0–10
    action_band: str                   # "hold"|"tighten_risk"|"staged_reduction"|"aggressive_exit"
    component_weights: Dict[str, float] = field(default_factory=dict)
    generated_at: str = ""
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def make_score_id(cls, symbol: str, scored_date: str) -> str:
        return _sha256(f"{symbol}|{scored_date}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExitVelocityScore":
        return cls(**{k: d.get(k, cls.__dataclass_fields__[k].default
                      if hasattr(cls.__dataclass_fields__[k].default, '__class__') else None)
                    for k in cls.__dataclass_fields__})


# ── Capital Lifecycle Layer ────────────────────────────────────────────────────

@dataclass
class PartialExitDecision:
    """Staged partial exit allocation decision (analytics output, no auto-execution)."""
    decision_id: str
    symbol: str
    decision_date: str
    velocity_score: float
    action_band: str
    stage_1_fraction: float            # default 0.33
    stage_2_fraction: float            # default 0.33
    stage_3_fraction: float            # default 0.34
    recommended_first_reduction: float # fraction to reduce now
    convexity_preservation_rationale: str
    generated_at: str = ""
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CounterfactualExitOutcome:
    """Post-exit counterfactual analytics (computed after n days pass)."""
    cf_id: str
    symbol: str
    actual_exit_date: str
    actual_exit_price: float
    realized_return: float
    # forward expectancy
    post_exit_return_3d: Optional[float]
    post_exit_return_5d: Optional[float]
    post_exit_return_10d: Optional[float]
    post_exit_return_20d: Optional[float]
    post_exit_max_excursion: Optional[float]   # best price after exit
    post_exit_max_drawdown: Optional[float]    # worst drawdown after exit
    counterfactual_convexity_retention: Optional[float]  # did price keep going?
    # comparison scenarios
    exit_was_premature: Optional[bool]
    exit_was_late: Optional[bool]
    generated_at: str = ""
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CounterfactualPartialExitRecord:
    """Full vs staged vs delayed vs accelerated exit comparison."""
    record_id: str
    symbol: str
    exit_date: str
    full_exit_return: float
    staged_exit_return: Optional[float]
    delayed_3d_return: Optional[float]
    accelerated_exit_return: Optional[float]
    best_scenario: str                 # "full"|"staged"|"delayed"|"accelerated"
    convexity_retained_by_staged: Optional[float]
    generated_at: str = ""
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LoserRetentionRecord:
    """Classify each losing trade by hold quality."""
    record_id: str
    symbol: str
    exit_date: str
    holding_days: int
    realized_return: float
    exit_velocity_score_at_exit: float
    # classification
    hold_classification: str           # "good_hold"|"bad_hold"|"good_cut"|"bad_cut"
    classification_rationale: str
    would_have_recovered: Optional[bool]   # 5d post-exit
    generated_at: str = ""
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConditionalExpectancyPoint:
    """Single cell in the conditional expectancy surface."""
    cell_id: str
    regime: str
    volatility_state: str              # "low"|"medium"|"high"
    rs_persistence: str                # "strong"|"moderate"|"weak"
    breadth_condition: str             # "expanding"|"neutral"|"contracting"
    deterioration_velocity: str        # "slow"|"moderate"|"fast"
    convexity_quality: str             # "high"|"medium"|"low"
    sector_leadership: str             # "leading"|"neutral"|"lagging"
    sample_count: int
    mean_forward_return_5d: Optional[float]
    mean_forward_return_10d: Optional[float]
    conditional_expectancy: Optional[float]
    generated_at: str = ""
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Research Dataset Layer ─────────────────────────────────────────────────────

@dataclass
class ExitResearchRecord:
    """Unified research dataset record — one row per completed position."""
    research_id: str                   # sha256(obs_id)
    obs_id: str
    symbol: str
    entry_date: str
    exit_date: str
    holding_days: int
    realized_return: float
    # deterioration state at exit
    exit_velocity_score: float
    action_band: str
    peak_decay_velocity: float
    structural_failure_score: float
    momentum_velocity: float
    convexity_retention_score: float
    persistence_score: float
    # regime context
    regime: str
    breadth_score: float
    rs_rank: float
    atr_pct: float
    # portfolio context
    concurrent_positions: int
    portfolio_heat: float
    # counterfactual
    post_exit_return_5d: Optional[float]
    post_exit_return_10d: Optional[float]
    exit_was_premature: Optional[bool]
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def make_research_id(cls, obs_id: str) -> str:
        return _sha256(f"research|{obs_id}")
