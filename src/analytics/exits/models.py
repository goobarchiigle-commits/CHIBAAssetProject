"""
src/analytics/exits/models.py — Exit convexity analytics dataclasses

Typed, deterministic, replay-compatible records.
No execution mutation. Analytics only.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

SCHEMA_VERSION: int = 1

# Exit reason constants
EXIT_REASON_TURTLE_55D = "turtle_exit_55d"
EXIT_REASON_STOP_LOSS = "stop_loss"
EXIT_REASON_TIME_EXIT = "time_exit"
EXIT_REASON_MANUAL = "manual"
EXIT_REASON_UNKNOWN = "unknown"

# Trailing policy names (counterfactual replay only)
POLICY_FIXED_TRAILING = "fixed_trailing"
POLICY_ATR_TRAILING = "atr_trailing"
POLICY_VOL_SCALED = "vol_scaled"
POLICY_TIME_DECAY = "time_decay"
POLICY_HYBRID = "hybrid"
POLICY_MOMENTUM_EXHAUSTION = "momentum_exhaustion"

ALL_POLICIES: List[str] = [
    POLICY_FIXED_TRAILING,
    POLICY_ATR_TRAILING,
    POLICY_VOL_SCALED,
    POLICY_TIME_DECAY,
    POLICY_HYBRID,
    POLICY_MOMENTUM_EXHAUSTION,
]


def _sha256(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_json(d: dict) -> str:
    return json.dumps(d, sort_keys=True, ensure_ascii=False)


@dataclass
class CompletedTrade:
    """Raw completed trade record. Input to exit analytics pipeline."""
    trade_id: str
    record_id: str                   # sha256(symbol|entry_ts|exit_ts)
    symbol: str
    entry_timestamp: str             # UTC ISO 8601
    exit_timestamp: str              # UTC ISO 8601
    holding_duration_days: float
    entry_price: float
    exit_price: float
    shares: int
    realized_return: float           # fraction: exit/entry - 1
    realized_pnl_jpy: float
    exit_reason: str
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def make_record_id(cls, symbol: str, entry_timestamp: str, exit_timestamp: str) -> str:
        return _sha256(f"{symbol}|{entry_timestamp}|{exit_timestamp}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HoldingPath:
    """Raw price/equity trajectory artifacts during hold. Append-only by design."""
    record_id: str                       # = CompletedTrade.record_id
    symbol: str
    entry_timestamp: str
    exit_timestamp: str
    holding_price_path: List[Dict[str, Any]]    # [{"day": float, "price": float}, ...]
    holding_equity_curve: List[Dict[str, Any]]  # [{"day": float, "pnl_jpy": float}, ...]
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HoldingPath":
        return cls(
            record_id=d["record_id"],
            symbol=d["symbol"],
            entry_timestamp=d["entry_timestamp"],
            exit_timestamp=d["exit_timestamp"],
            holding_price_path=d.get("holding_price_path", []),
            holding_equity_curve=d.get("holding_equity_curve", []),
            schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
        )


@dataclass
class PostExitContinuation:
    """Post-exit price continuation metrics. Enriched after exit date."""
    record_id: str
    post_exit_return_1d: Optional[float] = None
    post_exit_return_3d: Optional[float] = None
    post_exit_return_5d: Optional[float] = None
    benchmark_relative_post_exit_return_1d: Optional[float] = None
    benchmark_relative_post_exit_return_3d: Optional[float] = None
    benchmark_relative_post_exit_return_5d: Optional[float] = None
    sector_relative_post_exit_return: Optional[float] = None
    relative_trend_persistence_score: Optional[float] = None
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PostExitContinuation":
        return cls(
            record_id=d["record_id"],
            post_exit_return_1d=d.get("post_exit_return_1d"),
            post_exit_return_3d=d.get("post_exit_return_3d"),
            post_exit_return_5d=d.get("post_exit_return_5d"),
            benchmark_relative_post_exit_return_1d=d.get("benchmark_relative_post_exit_return_1d"),
            benchmark_relative_post_exit_return_3d=d.get("benchmark_relative_post_exit_return_3d"),
            benchmark_relative_post_exit_return_5d=d.get("benchmark_relative_post_exit_return_5d"),
            sector_relative_post_exit_return=d.get("sector_relative_post_exit_return"),
            relative_trend_persistence_score=d.get("relative_trend_persistence_score"),
            schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
        )


@dataclass
class ExitConvexityRecord:
    """Full exit convexity analytics record for a single completed trade."""
    record_id: str
    trade_id: str
    symbol: str
    entry_timestamp: str
    exit_timestamp: str
    holding_duration_days: float
    realized_return: float
    exit_reason: str

    # Excursion metrics
    max_favorable_excursion: float          # fraction from entry (>= 0)
    max_adverse_excursion: float            # fraction from entry (<= 0)
    peak_unrealized_pnl: float              # JPY
    pnl_giveback_ratio: float               # [0, 1]

    # Volatility metrics
    volatility_during_hold: float           # annualized daily vol
    volatility_adjusted_return: float       # realized_return / vol if vol > 0

    # Post-exit continuation
    post_exit_return_1d: Optional[float] = None
    post_exit_return_3d: Optional[float] = None
    post_exit_return_5d: Optional[float] = None
    benchmark_relative_post_exit_return_1d: Optional[float] = None
    benchmark_relative_post_exit_return_3d: Optional[float] = None
    benchmark_relative_post_exit_return_5d: Optional[float] = None
    sector_relative_post_exit_return: Optional[float] = None
    relative_trend_persistence_score: Optional[float] = None

    # Convexity diagnostics
    runner_continuation_score: float = 0.0
    convexity_capture_rate: float = 1.0
    runner_retention_rate: float = 1.0
    premature_exit_ratio: float = 0.0
    exit_efficiency_score: float = 1.0
    post_exit_alpha_leakage: Optional[float] = None
    trend_persistence_capture: Optional[float] = None

    # Holding-path analytics
    time_to_peak_pnl: float = 0.0           # days from entry
    exit_distance_from_peak: float = 0.0    # days: hold_duration - time_to_peak
    post_peak_decay_rate: float = 0.0       # fraction/day lost after peak
    trend_stability_score: float = 0.0      # [0, 1]

    # Diagnostic flags
    is_premature_exit: bool = False
    is_convexity_truncated: bool = False
    is_volatility_insensitive: bool = False
    is_runner_retained: bool = False

    generated_at: str = ""
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExitConvexityRecord":
        return cls(
            record_id=d["record_id"],
            trade_id=d.get("trade_id", ""),
            symbol=d["symbol"],
            entry_timestamp=d["entry_timestamp"],
            exit_timestamp=d["exit_timestamp"],
            holding_duration_days=float(d.get("holding_duration_days", 0.0)),
            realized_return=float(d.get("realized_return", 0.0)),
            exit_reason=d.get("exit_reason", EXIT_REASON_UNKNOWN),
            max_favorable_excursion=float(d.get("max_favorable_excursion", 0.0)),
            max_adverse_excursion=float(d.get("max_adverse_excursion", 0.0)),
            peak_unrealized_pnl=float(d.get("peak_unrealized_pnl", 0.0)),
            pnl_giveback_ratio=float(d.get("pnl_giveback_ratio", 0.0)),
            volatility_during_hold=float(d.get("volatility_during_hold", 0.0)),
            volatility_adjusted_return=float(d.get("volatility_adjusted_return", 0.0)),
            post_exit_return_1d=d.get("post_exit_return_1d"),
            post_exit_return_3d=d.get("post_exit_return_3d"),
            post_exit_return_5d=d.get("post_exit_return_5d"),
            benchmark_relative_post_exit_return_1d=d.get("benchmark_relative_post_exit_return_1d"),
            benchmark_relative_post_exit_return_3d=d.get("benchmark_relative_post_exit_return_3d"),
            benchmark_relative_post_exit_return_5d=d.get("benchmark_relative_post_exit_return_5d"),
            sector_relative_post_exit_return=d.get("sector_relative_post_exit_return"),
            relative_trend_persistence_score=d.get("relative_trend_persistence_score"),
            runner_continuation_score=float(d.get("runner_continuation_score", 0.0)),
            convexity_capture_rate=float(d.get("convexity_capture_rate", 1.0)),
            runner_retention_rate=float(d.get("runner_retention_rate", 1.0)),
            premature_exit_ratio=float(d.get("premature_exit_ratio", 0.0)),
            exit_efficiency_score=float(d.get("exit_efficiency_score", 1.0)),
            post_exit_alpha_leakage=d.get("post_exit_alpha_leakage"),
            trend_persistence_capture=d.get("trend_persistence_capture"),
            time_to_peak_pnl=float(d.get("time_to_peak_pnl", 0.0)),
            exit_distance_from_peak=float(d.get("exit_distance_from_peak", 0.0)),
            post_peak_decay_rate=float(d.get("post_peak_decay_rate", 0.0)),
            trend_stability_score=float(d.get("trend_stability_score", 0.0)),
            is_premature_exit=bool(d.get("is_premature_exit", False)),
            is_convexity_truncated=bool(d.get("is_convexity_truncated", False)),
            is_volatility_insensitive=bool(d.get("is_volatility_insensitive", False)),
            is_runner_retained=bool(d.get("is_runner_retained", False)),
            generated_at=d.get("generated_at", ""),
            schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
        )


@dataclass
class PolicyReplayResult:
    """Counterfactual trailing-policy replay result for one trade."""
    replay_id: str                           # sha256(trade_record_id|policy_name)
    policy_name: str
    trade_record_id: str
    symbol: str
    entry_timestamp: str
    actual_exit_timestamp: str
    hypothetical_exit_timestamp: str
    actual_realized_return: float
    hypothetical_realized_return: float
    convexity_retention: float               # hyp_return / MFE, clipped [0, 1]
    pnl_giveback: float                      # fraction of peak given back at hyp exit
    runner_retention: float                  # hyp_return / actual_return if actual > 0
    generated_at: str = ""
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def make_replay_id(cls, trade_record_id: str, policy_name: str) -> str:
        return _sha256(f"{trade_record_id}|{policy_name}")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PolicyReplayResult":
        return cls(
            replay_id=d["replay_id"],
            policy_name=d["policy_name"],
            trade_record_id=d["trade_record_id"],
            symbol=d["symbol"],
            entry_timestamp=d["entry_timestamp"],
            actual_exit_timestamp=d["actual_exit_timestamp"],
            hypothetical_exit_timestamp=d["hypothetical_exit_timestamp"],
            actual_realized_return=float(d["actual_realized_return"]),
            hypothetical_realized_return=float(d["hypothetical_realized_return"]),
            convexity_retention=float(d["convexity_retention"]),
            pnl_giveback=float(d["pnl_giveback"]),
            runner_retention=float(d["runner_retention"]),
            generated_at=d.get("generated_at", ""),
            schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
        )


@dataclass
class RunnerRetentionSummary:
    """Aggregate diagnostics across a batch of ExitConvexityRecords."""
    total_trades: int
    premature_exit_count: int
    convexity_truncated_count: int
    volatility_insensitive_count: int
    runner_retained_count: int
    premature_exit_rate: float
    convexity_truncation_rate: float
    mean_convexity_capture_rate: float
    mean_exit_efficiency_score: float
    mean_pnl_giveback_ratio: float
    mean_runner_retention_rate: float
    mean_post_exit_alpha_leakage: Optional[float]
    top_premature_exits: List[str]          # record_ids sorted by post_exit_return_5d desc
    top_convexity_truncations: List[str]    # record_ids sorted by convexity_capture_rate asc
    generated_at: str = ""
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
