"""
src/analytics/exits — Exit convexity observability and exit-efficiency analytics.

Modules:
  models          — typed dataclasses (CompletedTrade, HoldingPath, ExitConvexityRecord, ...)
  convexity       — ExitConvexityAnalyzer, RunnerRetentionDiagnostics, ExitEfficiencyScorer
  continuation    — PostExitContinuationTracker, VolatilityAdjustedExitAnalytics
  trailing_replay — counterfactual trailing-policy replayer
  reports         — JSONL / parquet / markdown persistence

Analytics only. No execution mutation. Deterministic outputs.
"""
from src.analytics.exits.models import (
    ALL_POLICIES,
    POLICY_ATR_TRAILING,
    POLICY_FIXED_TRAILING,
    POLICY_HYBRID,
    POLICY_MOMENTUM_EXHAUSTION,
    POLICY_TIME_DECAY,
    POLICY_VOL_SCALED,
    CompletedTrade,
    ExitConvexityRecord,
    HoldingPath,
    PolicyReplayResult,
    PostExitContinuation,
    RunnerRetentionSummary,
)
from src.analytics.exits.convexity import (
    ExitConvexityAnalyzer,
    ExitEfficiencyScorer,
    RunnerRetentionDiagnostics,
)
from src.analytics.exits.continuation import (
    PostExitContinuationTracker,
    VolatilityAdjustedExitAnalytics,
)
from src.analytics.exits.trailing_replay import TrailingPolicyReplayer
from src.analytics.exits.reports import (
    append_exit_record,
    append_policy_replay,
    generate_markdown_report,
    load_exit_records,
    load_holding_path,
    load_policy_replays,
    persist_holding_path,
    write_exit_records_parquet,
    write_policy_replays_parquet,
    write_report_atomic,
)

__all__ = [
    # models
    "CompletedTrade", "HoldingPath", "PostExitContinuation",
    "ExitConvexityRecord", "PolicyReplayResult", "RunnerRetentionSummary",
    "ALL_POLICIES", "POLICY_FIXED_TRAILING", "POLICY_ATR_TRAILING",
    "POLICY_VOL_SCALED", "POLICY_TIME_DECAY", "POLICY_HYBRID",
    "POLICY_MOMENTUM_EXHAUSTION",
    # convexity
    "ExitConvexityAnalyzer", "ExitEfficiencyScorer", "RunnerRetentionDiagnostics",
    # continuation
    "PostExitContinuationTracker", "VolatilityAdjustedExitAnalytics",
    # trailing replay
    "TrailingPolicyReplayer",
    # reports
    "append_exit_record", "append_policy_replay", "generate_markdown_report",
    "load_exit_records", "load_holding_path", "load_policy_replays",
    "persist_holding_path", "write_exit_records_parquet",
    "write_policy_replays_parquet", "write_report_atomic",
]
