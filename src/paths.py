"""
paths.py — プロジェクトパス・ライブ安全設定 一元管理
最終更新: 2026-03-31（C:/ai-trading 移行対応）

設計原則:
  1. PROJECT_ROOT  — このファイル（paths.py）の場所 = コードベースの起点
  2. BASE_DIR      — C:/ai-trading（外部データの単一正本）
                     AI_TRADING_HOME 環境変数で上書き可能（.env で設定推奨）
  3. ライブ安全ガード — LIVE_MODE / MAX_ORDERS_PER_DAY を環境変数で制御

環境変数一覧:
    AI_TRADING_HOME      外部データ基点（デフォルト: C:/ai-trading）
    LIVE_UNIVERSE_FILE   ユニバースファイルパス（PROJECT_ROOT 相対 or 絶対パス）
    LIVE_MODE            true にすると実発注モード（デフォルト: false）
    MAX_ORDERS_PER_DAY   1日の最大発注件数（デフォルト: 20）

使い方:
    from src.paths import PROJECT_ROOT, BASE_DIR, RESULTS_DIR, SIGNALS_DIR
    from src.paths import LIVE_MODE, assert_live_ready
"""
from __future__ import annotations

from functools import lru_cache
import json
import os
import time
from pathlib import Path


# ────────────────────────────────────────────────────────────────────────────
# PROJECT_ROOT — コードベースの起点（このファイルの場所）
# ────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent
SRC_DIR: Path = PROJECT_ROOT


# ────────────────────────────────────────────────────────────────────────────
# .env 読み込み（python-dotenv があれば使用、なければ手動パース）
# paths を import するだけで .env が環境変数に反映される
# ────────────────────────────────────────────────────────────────────────────
_env_file = PROJECT_ROOT / ".env"
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(_env_file)
except ImportError:
    if _env_file.exists():
        for _line in _env_file.read_text(encoding="utf-8").splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())


# ────────────────────────────────────────────────────────────────────────────
# BASE_DIR — 外部データの単一正本（C:/ai-trading）
#
# AI_TRADING_HOME 環境変数（.env に記載）で上書き可能。
# 未設定の場合は PROJECT_ROOT（コードと同じ場所）にフォールバックするが、
# 通常は .env の AI_TRADING_HOME=C:/ai-trading が有効になる。
# ────────────────────────────────────────────────────────────────────────────
def _env_str(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int, minimum: int | None = None) -> int:
    raw = os.environ.get(name, "")
    if raw == "":
        value = default
    else:
        try:
            value = int(raw)
        except ValueError as exc:
            raise RuntimeError(
                f"環境変数 {name}='{raw}' は整数ではありません。"
            ) from exc
    if minimum is not None and value < minimum:
        raise RuntimeError(
            f"環境変数 {name}={value} は不正です。{minimum} 以上を指定してください。"
        )
    return value


def _env_float(name: str, default: float, minimum: float | None = None) -> float:
    raw = os.environ.get(name, "")
    if raw == "":
        value = default
    else:
        try:
            value = float(raw)
        except ValueError as exc:
            raise RuntimeError(
                f"環境変数 {name}='{raw}' は数値ではありません。"
            ) from exc
    if minimum is not None and value < minimum:
        raise RuntimeError(
            f"環境変数 {name}={value} は不正です。{minimum} 以上を指定してください。"
        )
    return value


BASE_DIR: Path = Path(_env_str("AI_TRADING_HOME", str(PROJECT_ROOT)))

# ────────────────────────────────────────────────────────────────────────────
# 外部データディレクトリ（BASE_DIR 配下 = C:/ai-trading 配下）
# ────────────────────────────────────────────────────────────────────────────
RESULTS_DIR:  Path = BASE_DIR / "backtests"
LOGS_DIR:     Path = BASE_DIR / "logs"
CACHE_DIR:    Path = BASE_DIR / "cache"
DATASET_DIR:  Path = BASE_DIR / "data"
RUNTIME_DIR:  Path = BASE_DIR / "runtime"
REPORTS_DIR:  Path = BASE_DIR / "reports"
DOCS_DIR:     Path = BASE_DIR / "docs" / "research"


# ────────────────────────────────────────────────────────────────────────────
# コードと一体のディレクトリ（PROJECT_ROOT 相対・上書き不可）
# ────────────────────────────────────────────────────────────────────────────
CONFIGS_DIR:  Path = PROJECT_ROOT / "configs"
UNIVERSE_DIR: Path = CONFIGS_DIR  / "universe"
STRATEGY_CONFIG_FILE: Path = CONFIGS_DIR / "strategy.yaml"


# ────────────────────────────────────────────────────────────────────────────
# データパイプライン
# ────────────────────────────────────────────────────────────────────────────
BACKTEST_DATASET_DIR: Path = DATASET_DIR / "backtest_dataset"
SIGNALS_DIR:          Path = DATASET_DIR / "signals"


def latest_dataset_version() -> str:
    """利用可能な backtest snapshot の最新バージョンを返す。"""
    if not BACKTEST_DATASET_DIR.exists():
        return ""
    versions = sorted(
        p.name for p in BACKTEST_DATASET_DIR.iterdir()
        if p.is_dir() and (p / "_meta.json").exists()
    )
    return versions[-1] if versions else ""


def load_strategy_config() -> dict:
    """後方互換: 新しい config_loader に委譲する。"""
    from src.config_loader import load_strategy_config as _load_strategy_config

    return _load_strategy_config()


# ────────────────────────────────────────────────────────────────────────────
# 設定ファイル（環境変数で上書き可能）
# ────────────────────────────────────────────────────────────────────────────
RSR_UNIVERSE_FILE: Path = CONFIGS_DIR / "rsr_universe_42.csv"
RSR_SNAPSHOT_DIR:  Path = RUNTIME_DIR / "rsr"          # daily RSR snapshots (append-only)
DEMOTION_PRESSURE_FILE: Path = RUNTIME_DIR / "demotion_pressure.json"
# ── Demotion Actuation layer (Phase 3A) ───────────────────────────────────────
PENDING_DEMOTION_FILE:            Path = RUNTIME_DIR / "pending_demotion.jsonl"
DEMOTION_ACTUATION_METRICS_FILE:  Path = RUNTIME_DIR / "demotion_actuation_metrics.jsonl"
OHLCV_DIR:         Path = BASE_DIR / "data" / "ohlcv"  # per-symbol OHLCV CSV cache
UNIVERSE_DETERMINISM_AUDIT_DIR: Path = RUNTIME_DIR / "determinism" / "universe_audit"

def _resolve_path(env_var: str, default: Path) -> Path:
    val = os.environ.get(env_var, "").strip()
    if not val:
        return default
    p = Path(val)
    return p if p.is_absolute() else PROJECT_ROOT / p

LIVE_UNIVERSE_FILE:   Path = _resolve_path(
    "LIVE_UNIVERSE_FILE", UNIVERSE_DIR / "rsr42_trading.json"
)
SHADOW_UNIVERSE_FILE: Path = UNIVERSE_DIR / "shadow_universe.json"


# ────────────────────────────────────────────────────────────────────────────
# ライブ運用ファイル
# ────────────────────────────────────────────────────────────────────────────
ORDER_LOCK_FILE:     Path = RUNTIME_DIR / "order_lock.json"       # duplicate-order idempotency ledger
EXECUTION_LOCK_FILE: Path = RUNTIME_DIR / "execution.lock.json"   # execution mutex only
LIVE_LOG_DIR:        Path = LOGS_DIR    / "live"
PHASE2_METRICS_FILE: Path = LOGS_DIR    / "phase2_live_metrics.jsonl"

# ── Deferred entry observability layer ───────────────────────────────────────
DEFERRED_QUEUE_FILE:          Path = RUNTIME_DIR / "deferred_entries.json"
DEFERRED_METRICS_FILE:        Path = RUNTIME_DIR / "deferred_metrics.jsonl"
MISSED_ENTRY_TRACKING_FILE:   Path = RUNTIME_DIR / "missed_entry_tracking.jsonl"

# ── Execution lock + deployability observability ──────────────────────────────
LOCK_RECOVERY_LOG_FILE:       Path = RUNTIME_DIR / "lock_recovery.jsonl"
DEPLOYABILITY_METRICS_FILE:   Path = RUNTIME_DIR / "deployability_metrics.jsonl"
WATCHDOG_LOG_FILE:            Path = RUNTIME_DIR / "watchdog.jsonl"
ORDERS_JOURNAL_DIR:           Path = RUNTIME_DIR / "orders"
INFLIGHT_REGISTRY_FILE:       Path = RUNTIME_DIR / "inflight_orders.jsonl"

# ── Runtime governance reports ────────────────────────────────────────────────
RUNTIME_SUMMARY_DIR:    Path = REPORTS_DIR / "runtime_summary"
EXECUTION_HEALTH_DIR:   Path = REPORTS_DIR / "execution_health"
GOVERNANCE_SUMMARY_DIR: Path = REPORTS_DIR / "governance_summary"

# ── Capital scaling layer (Phase 1: capital injection ramp) ──────────────────
CAPITAL_STATE_FILE:          Path = RUNTIME_DIR / "capital_state.json"
CAPITAL_TELEMETRY_FILE:      Path = RUNTIME_DIR / "capital_scaling_telemetry.jsonl"
CAPITAL_FREEZE_STATE_FILE:   Path = RUNTIME_DIR / "capital_freeze_state.json"
DEPLOYMENT_RAMP_STATE_FILE:  Path = RUNTIME_DIR / "deployment_ramp_state.json"
SHADOW_SIMULATION_FILE:      Path = RUNTIME_DIR / "shadow_simulation.jsonl"

# ── Adaptive growth layer (Phase 2: geometric-growth optimizer) ───────────────
AGGRESSION_STATE_FILE:        Path = RUNTIME_DIR / "aggression_state.json"
EDGE_MODEL_STATE_FILE:        Path = RUNTIME_DIR / "edge_model_state.json"
DEPLOYMENT_STATE_RECORD_FILE: Path = RUNTIME_DIR / "deployment_state_record.json"
MULTIHORIZON_STATE_FILE:      Path = RUNTIME_DIR / "multihorizon_confidence.json"
CONCENTRATION_METRICS_FILE:   Path = RUNTIME_DIR / "concentration_metrics.json"
EXPLORATION_STATE_FILE:       Path = RUNTIME_DIR / "exploration_state.json"
OPPORTUNITY_COST_LOG_FILE:    Path = RUNTIME_DIR / "opportunity_cost.jsonl"
REFLEXIVITY_STATE_FILE:       Path = RUNTIME_DIR / "reflexivity_state.json"
REFLEXIVITY_LOG_FILE:         Path = RUNTIME_DIR / "reflexivity_observations.jsonl"

# ── Portfolio truth layer (ground truth under execution uncertainty) ──────────
TRUTH_DIR:                 Path = RUNTIME_DIR / "truth"
EXECUTION_TRUTH_FILE:      Path = TRUTH_DIR / "execution_records.jsonl"
INTENT_SNAPSHOT_FILE:      Path = TRUTH_DIR / "intent_snapshots.jsonl"
POSITION_TRUTH_FILE:       Path = TRUTH_DIR / "position_truth.json"
CAPITAL_TRUTH_FILE:        Path = TRUTH_DIR / "capital_truth.json"
TRUTH_TELEMETRY_FILE:      Path = TRUTH_DIR / "truth_telemetry.jsonl"

# ── Economic exposure + capital efficiency layer (Phase 5A) ──────────────────
ALLOCATION_DIR:               Path = RUNTIME_DIR / "allocation"
EXPOSURE_STATE_FILE:          Path = ALLOCATION_DIR / "exposure_state.json"
ALLOCATION_DECISIONS_FILE:    Path = ALLOCATION_DIR / "allocation_decisions.jsonl"
ALPHA_LIFECYCLE_DIR:          Path = ALLOCATION_DIR / "alpha_lifecycle"
ALLOCATOR_STABILITY_FILE:     Path = ALLOCATION_DIR / "allocator_stability.jsonl"
STABILITY_WINDOW_FILE:        Path = ALLOCATION_DIR / "stability_window.json"
COMPLEXITY_BUDGET_FILE:       Path = ALLOCATION_DIR / "complexity_budget.json"
COMPLEXITY_AUDIT_FILE:        Path = ALLOCATION_DIR / "complexity_audit.jsonl"
ALLOCATION_INTENT_FILE:       Path = ALLOCATION_DIR / "allocation_intent.jsonl"
SKIPPED_TRADE_FILE:           Path = ALLOCATION_DIR / "skipped_trades.jsonl"
SKIPPED_OUTCOMES_FILE:        Path = ALLOCATION_DIR / "skipped_trade_outcomes.jsonl"
ALLOCATION_OUTCOMES_FILE:     Path = ALLOCATION_DIR / "allocation_outcomes.jsonl"
OUTCOMES_ARCHIVE_DIR:         Path = ALLOCATION_DIR / "archive"
STALE_VISIBILITY_FILE:        Path = ALLOCATION_DIR / "stale_visibility.jsonl"

# ── Autonomous recovery layer ─────────────────────────────────────────────────
BOOTSTRAP_RECOVERY_FILE: Path = RUNTIME_DIR / "bootstrap_recovery.jsonl"
RUNTIME_HEALTH_FILE:     Path = RUNTIME_DIR / "runtime_health.jsonl"
MARKET_DATA_GUARD_FILE:  Path = RUNTIME_DIR / "market_data_guard.jsonl"
DISK_GUARD_FILE:         Path = RUNTIME_DIR / "disk_guard.jsonl"
CORRUPTION_AUDIT_FILE:   Path = RUNTIME_DIR / "corruption_audit.jsonl"

# ── Broker reconciliation layer ───────────────────────────────────────────────
BROKER_SNAPSHOT_FILE:       Path = RUNTIME_DIR / "broker_truth_snapshots.jsonl"
RECONCILIATION_LOG_FILE:    Path = RUNTIME_DIR / "reconciliation_log.jsonl"
EXECUTION_INTEGRITY_FILE:   Path = RUNTIME_DIR / "execution_integrity_log.jsonl"
REPLAY_CONSISTENCY_FILE:    Path = RUNTIME_DIR / "replay_consistency_log.jsonl"

# ── Execution lineage + epoch governance layer ────────────────────────────────
LINEAGE_DIR:                  Path = RUNTIME_DIR / "lineage"
EXECUTION_EPOCH_FILE:         Path = LINEAGE_DIR / "reconciliation_epochs.jsonl"
DEPLOYMENT_EPOCH_FILE:        Path = LINEAGE_DIR / "deployment_epochs.jsonl"
AUTHORITY_CHAIN_FILE:         Path = LINEAGE_DIR / "authority_chains.jsonl"
DEPLOYMENT_LINEAGE_FILE:      Path = LINEAGE_DIR / "deployment_lineage.jsonl"
DEPLOYMENT_MANIFEST_FILE:     Path = LINEAGE_DIR / "deployment_manifests.jsonl"
PENDING_AUTHORITY_FILE:       Path = LINEAGE_DIR / "pending_order_authority.jsonl"
AUTHORITY_EVENTS_FILE:        Path = LINEAGE_DIR / "authority_transition_events.jsonl"
QUARANTINE_ENTRIES_FILE:      Path = LINEAGE_DIR / "quarantine_entries.jsonl"
QUARANTINE_EXITS_FILE:        Path = LINEAGE_DIR / "quarantine_exits.jsonl"

# ── Deployable-alpha leakage analytics layer ──────────────────────────────────
ANALYTICS_DIR:                Path = RUNTIME_DIR / "analytics"
SKIPPED_OPPORTUNITY_FILE:     Path = ANALYTICS_DIR / "skipped_opportunities.jsonl"
ALPHA_METRICS_FILE:           Path = ANALYTICS_DIR / "alpha_metrics.jsonl"
DAILY_LEAKAGE_FILE:           Path = ANALYTICS_DIR / "daily_leakage_summaries.jsonl"
ALPHA_DIAGNOSTIC_FILE:        Path = ANALYTICS_DIR / "alpha_diagnostics.jsonl"
ALPHA_DIAGNOSTIC_REPORT_DIR:  Path = ANALYTICS_DIR / "reports"
RESEARCH_PRIORITY_FILE:       Path = ANALYTICS_DIR / "research_priority_summaries.jsonl"
RESEARCH_PRIORITY_REPORT_DIR: Path = RUNTIME_DIR / "reports"

# ── Weekly market intelligence layer ─────────────────────────────────────────
WEEKLY_INTELLIGENCE_REPORT_DIR:  Path = RUNTIME_DIR / "reports"
WEEKLY_INTELLIGENCE_ARCHIVE_FILE: Path = RUNTIME_DIR / "reports" / "weekly_intelligence_archive.jsonl"
WEEKLY_INTELLIGENCE_CHARTS_DIR:  Path = RUNTIME_DIR / "reports" / "charts"

# ── Feature forward expectancy layer ─────────────────────────────────────────
FEATURE_SNAPSHOTS_DIR:       Path = RUNTIME_DIR / "feature_snapshots"
ENTRY_FEATURES_FILE:         Path = FEATURE_SNAPSHOTS_DIR / "entry_features.jsonl"
POSITION_LIFECYCLE_FILE:     Path = FEATURE_SNAPSHOTS_DIR / "position_lifecycle.jsonl"
FORWARD_RETURN_CACHE_DIR:    Path = FEATURE_SNAPSHOTS_DIR / "forward_return_cache"

# ── Runtime integration stabilization layer ──────────────────────────────────
INTEGRATION_DIR:                     Path = RUNTIME_DIR / "integration"
INTEGRATION_MANIFEST_FILE:           Path = INTEGRATION_DIR / "dependency_manifests.jsonl"
INTEGRATION_LINEAGE_FILE:            Path = INTEGRATION_DIR / "initialization_lineage.jsonl"
INTEGRATION_STATE_REPORT_FILE:       Path = INTEGRATION_DIR / "state_validation_reports.jsonl"
INTEGRATION_COHERENCE_FILE:          Path = INTEGRATION_DIR / "runtime_coherence_diagnostics.jsonl"
INTEGRATION_FAILURE_FILE:            Path = INTEGRATION_DIR / "integration_failure_attribution.jsonl"
INTEGRATION_BROKER_CONSISTENCY_FILE: Path = INTEGRATION_DIR / "broker_runtime_consistency_reports.jsonl"

# ── Deployable universe governance layer ─────────────────────────────────────
UNIVERSE_RUNTIME_DIR:          Path = RUNTIME_DIR / "universe"
UNIVERSE_MANIFEST_FILE:        Path = UNIVERSE_RUNTIME_DIR / "universe_manifest.jsonl"
UNIVERSE_PROMOTION_FILE:       Path = UNIVERSE_RUNTIME_DIR / "promotion_decisions.jsonl"
UNIVERSE_LINEAGE_FILE:         Path = UNIVERSE_RUNTIME_DIR / "universe_lineage.jsonl"
UNIVERSE_DIAGNOSTICS_FILE:     Path = UNIVERSE_RUNTIME_DIR / "deployability_diagnostics.jsonl"
UNIVERSE_GOVERNANCE_REPORT_DIR: Path = UNIVERSE_RUNTIME_DIR / "universe_governance_reports"
# AUTO_PROMOTE_SAFE_V2: probation-based deployment tracking
PROBATION_PROMOTIONS_FILE:     Path = UNIVERSE_RUNTIME_DIR / "probation_promotions.jsonl"
PROBATION_OUTCOMES_FILE:       Path = UNIVERSE_RUNTIME_DIR / "probation_outcomes.jsonl"
PROBATION_REJECTION_FILE:      Path = UNIVERSE_RUNTIME_DIR / "promotion_rejections.jsonl"
# ── Universe self-healing (Shrink Engine + Shadow Refill) ────────────────────
UNIVERSE_SHRINK_LOG_FILE: Path = UNIVERSE_RUNTIME_DIR / "universe_shrink_candidates.jsonl"
SHADOW_REFILL_LOG_FILE:   Path = UNIVERSE_RUNTIME_DIR / "shadow_refill.jsonl"
# ── Rotation Planner layer (Phase 3B) ─────────────────────────────────────────
ROTATION_PLAN_FILE:            Path = UNIVERSE_RUNTIME_DIR / "rotation_plan.json"
ROTATION_PLAN_HISTORY_FILE:    Path = UNIVERSE_RUNTIME_DIR / "rotation_plan_history.jsonl"
# ── Governance Health Engine (Phase 3C) ───────────────────────────────────────
GOVERNANCE_HEALTH_FILE:        Path = UNIVERSE_RUNTIME_DIR / "governance_health.json"
GOVERNANCE_MODE_HISTORY_FILE:  Path = UNIVERSE_RUNTIME_DIR / "governance_mode_history.jsonl"
GOVERNANCE_OVERRIDE_FILE:      Path = UNIVERSE_RUNTIME_DIR / "governance_override.json"
AUTO_FREEZE_STATE_FILE:        Path = UNIVERSE_RUNTIME_DIR / "auto_freeze_state.json"
# ── Decision Context Capture (Phase 4C-lite) ─────────────────────────────────
DECISION_CONTEXT_HISTORY_FILE: Path = UNIVERSE_RUNTIME_DIR / "decision_context_history.jsonl"
# Breakout Quality Intelligence: daily candle quality telemetry
BREAKOUT_QUALITY_TELEMETRY_FILE: Path = RUNTIME_DIR / "analytics" / "breakout_quality_telemetry.jsonl"
# Continuation Priority Intelligence: held-position ranking telemetry
CONTINUATION_PRIORITY_TELEMETRY_FILE: Path = RUNTIME_DIR / "analytics" / "continuation_priority_telemetry.jsonl"
# Entry Timing Intelligence: BUY candidate entry quality telemetry
ENTRY_TIMING_TELEMETRY_FILE: Path = RUNTIME_DIR / "analytics" / "entry_timing_telemetry.jsonl"
# Entry Timing Promotion: tier state + history
ENTRY_TIMING_REPORTS_DIR:    Path = REPORTS_DIR / "entry_timing"
ENTRY_TIMING_PROMOTION_FILE: Path = ENTRY_TIMING_REPORTS_DIR / "entry_timing_promotion.json"
ENTRY_TIMING_HISTORY_FILE:   Path = ENTRY_TIMING_REPORTS_DIR / "entry_timing_history.csv"
# Position Sizing Intelligence: virtual conviction weight telemetry
POSITION_SIZING_TELEMETRY_FILE: Path = RUNTIME_DIR / "analytics" / "position_sizing_telemetry.jsonl"
# Position Sizing Promotion: tier state + history
POSITION_SIZING_REPORTS_DIR:    Path = REPORTS_DIR / "position_sizing"
POSITION_SIZING_PROMOTION_FILE: Path = POSITION_SIZING_REPORTS_DIR / "position_sizing_promotion.json"
POSITION_SIZING_HISTORY_FILE:   Path = POSITION_SIZING_REPORTS_DIR / "position_sizing_history.csv"

# ── Opportunity Cost Intelligence ────────────────────────────────────────────
OPPORTUNITY_COST_OC_DIR:              Path = ANALYTICS_DIR / "opportunity_cost"
OPPORTUNITY_COST_EVENTS_FILE:         Path = OPPORTUNITY_COST_OC_DIR / "opportunity_cost_events.jsonl"
OPPORTUNITY_COST_MATERIALIZED_FILE:   Path = OPPORTUNITY_COST_OC_DIR / "opportunity_cost_materialized.jsonl"
OC_5D_ENRICHED_FILE:                  Path = OPPORTUNITY_COST_OC_DIR / "oc_5d_enriched.jsonl"

# ── Slot Pressure Forecasting (Phase 5E) ──────────────────────────────────────
SLOT_PRESSURE_DIR:             Path = ANALYTICS_DIR / "slot_pressure"
SLOT_PRESSURE_FILE:            Path = SLOT_PRESSURE_DIR / "slot_pressure.jsonl"
SLOT_PRESSURE_ENRICHED_FILE:   Path = SLOT_PRESSURE_DIR / "slot_pressure_enriched.jsonl"

# ── Forward Continuation Outcome Intelligence ─────────────────────────────────
FORWARD_CONTINUATION_OUTCOME_DIR:  Path = ANALYTICS_DIR / "forward_continuation_outcome"
FORWARD_CONTINUATION_OUTCOME_FILE: Path = FORWARD_CONTINUATION_OUTCOME_DIR / "fco_aggregated.jsonl"

# ── Signal Attribution Intelligence ──────────────────────────────────────────
SIGNAL_ATTRIBUTION_DIR:  Path = ANALYTICS_DIR / "signal_attribution"
SIGNAL_ATTRIBUTION_FILE: Path = SIGNAL_ATTRIBUTION_DIR / "signal_attribution.jsonl"

# ── Capital Concentration Shadow Intelligence ─────────────────────────────────
CAPITAL_CONCENTRATION_SHADOW_DIR:  Path = ANALYTICS_DIR / "capital_concentration_shadow"
CAPITAL_CONCENTRATION_SHADOW_FILE: Path = CAPITAL_CONCENTRATION_SHADOW_DIR / "capital_concentration_shadow.jsonl"

# ── Priority Calibration Intelligence ────────────────────────────────────────
PRIORITY_CALIBRATION_DIR:  Path = ANALYTICS_DIR / "priority_calibration"
PRIORITY_CALIBRATION_FILE: Path = PRIORITY_CALIBRATION_DIR / "priority_calibration.jsonl"

# ── Hold Duration Calibration Intelligence ────────────────────────────────────
HOLD_DURATION_CALIBRATION_DIR:  Path = ANALYTICS_DIR / "hold_duration_calibration"
HOLD_DURATION_CALIBRATION_FILE: Path = HOLD_DURATION_CALIBRATION_DIR / "hold_duration_calibration.jsonl"

# ── Evidence Promotion Engine ─────────────────────────────────────────────────
EVIDENCE_PROMOTION_DIR:  Path = ANALYTICS_DIR / "evidence_promotion"
EVIDENCE_PROMOTION_FILE: Path = EVIDENCE_PROMOTION_DIR / "evidence_promotion.jsonl"

# ── Shadow Recommendation Engine ──────────────────────────────────────────────
SHADOW_RECOMMENDATION_DIR:  Path = ANALYTICS_DIR / "shadow_recommendation"
SHADOW_RECOMMENDATION_FILE: Path = SHADOW_RECOMMENDATION_DIR / "shadow_recommendation.jsonl"

# ── Shadow Outcome Tracker (Phase 6A-Lite) ────────────────────────────────────
SHADOW_OUTCOME_DIR:  Path = ANALYTICS_DIR / "shadow_outcome"
SHADOW_OUTCOME_FILE: Path = SHADOW_OUTCOME_DIR / "shadow_outcome.jsonl"

# ── Promotion Readiness Engine (Phase 6B-Lite) ────────────────────────────────
PROMOTION_READINESS_DIR:  Path = ANALYTICS_DIR / "promotion_readiness"
PROMOTION_READINESS_FILE: Path = PROMOTION_READINESS_DIR / "promotion_readiness.jsonl"

# ── Promotion Candidate Registry (Phase 6C-Lite) ──────────────────────────────
PROMOTION_CANDIDATE_REGISTRY_DIR:  Path = ANALYTICS_DIR / "promotion_candidate_registry"
PROMOTION_CANDIDATE_REGISTRY_FILE: Path = PROMOTION_CANDIDATE_REGISTRY_DIR / "promotion_candidate_registry.jsonl"

# ── Promotion Deployment Planner (Phase 6D-Lite) ──────────────────────────────
PROMOTION_DEPLOYMENT_PLAN_DIR:  Path = ANALYTICS_DIR / "promotion_deployment_plan"
PROMOTION_DEPLOYMENT_PLAN_FILE: Path = PROMOTION_DEPLOYMENT_PLAN_DIR / "promotion_deployment_plan.jsonl"

# ── Shadow Deployment Simulator (Phase 6E-Lite) ───────────────────────────────
SHADOW_DEPLOYMENT_DIR:  Path = ANALYTICS_DIR / "shadow_deployment"
SHADOW_DEPLOYMENT_FILE: Path = SHADOW_DEPLOYMENT_DIR / "shadow_deployment.jsonl"

# ── Promotion Impact Tracker (Phase 6F-Lite) ──────────────────────────────────
PROMOTION_IMPACT_TRACKER_DIR:  Path = ANALYTICS_DIR / "promotion_impact_tracker"
PROMOTION_IMPACT_TRACKER_FILE: Path = PROMOTION_IMPACT_TRACKER_DIR / "promotion_impact_tracker.jsonl"

# ── Promotion Governance Scoreboard (Phase 6G-Lite) ───────────────────────────
PROMOTION_GOVERNANCE_SCOREBOARD_DIR:  Path = ANALYTICS_DIR / "promotion_governance_scoreboard"
PROMOTION_GOVERNANCE_SCOREBOARD_FILE: Path = PROMOTION_GOVERNANCE_SCOREBOARD_DIR / "promotion_governance_scoreboard.jsonl"

# ── Promotion Rollout Manager (Phase 6H-Lite) ─────────────────────────────────
PROMOTION_ROLLOUT_MANAGER_DIR:  Path = ANALYTICS_DIR / "promotion_rollout_manager"
PROMOTION_ROLLOUT_MANAGER_FILE: Path = PROMOTION_ROLLOUT_MANAGER_DIR / "promotion_rollout_manager.jsonl"

# ── Production Promotion Gate (Phase 6I-Lite) ─────────────────────────────────
PRODUCTION_PROMOTION_GATE_DIR:  Path = ANALYTICS_DIR / "production_promotion_gate"
PRODUCTION_PROMOTION_GATE_FILE: Path = PRODUCTION_PROMOTION_GATE_DIR / "production_promotion_gate.jsonl"

# ── Active Experiment Registry (Phase 6J-Plus) ────────────────────────────────
EXPERIMENTS_DIR:                   Path = ANALYTICS_DIR / "experiments"
ACTIVE_EXPERIMENT_REGISTRY_FILE:   Path = EXPERIMENTS_DIR / "active_experiment_registry.json"
EXPERIMENT_ASSIGNMENT_FILE:        Path = EXPERIMENTS_DIR / "experiment_assignments.jsonl"
EXPERIMENT_PERFORMANCE_FILE:       Path = EXPERIMENTS_DIR / "experiment_performance.jsonl"

# ── Automatic Rollback Governance (Phase 6L-Lite) ─────────────────────────────
ROLLBACK_GOVERNANCE_DIR:  Path = ANALYTICS_DIR / "rollback_governance"
ROLLBACK_GOVERNANCE_FILE: Path = ROLLBACK_GOVERNANCE_DIR / "rollback_governance.jsonl"

# ── Progressive Capital Escalation (Phase 6M) ─────────────────────────────────
CAPITAL_ESCALATION_DIR:  Path = ANALYTICS_DIR / "capital_escalation"
CAPITAL_ESCALATION_FILE: Path = CAPITAL_ESCALATION_DIR / "capital_escalation.jsonl"

# ── Improvement Attribution Ranking (Phase 6N) ───────────────────────────────
IMPROVEMENT_RANKING_DIR:  Path = ANALYTICS_DIR / "improvement_ranking"
IMPROVEMENT_RANKING_FILE: Path = IMPROVEMENT_RANKING_DIR / "improvement_ranking.jsonl"

# ── System Health Audit (Phase 7A) ────────────────────────────────────────────
SYSTEM_HEALTH_DIR:         Path = ANALYTICS_DIR / "system_health"
SYSTEM_HEALTH_REPORT_FILE: Path = SYSTEM_HEALTH_DIR / "system_health_report.json"

# ── Weekly Executive Review (Phase 7B) ───────────────────────────────────────
WEEKLY_EXECUTIVE_REVIEW_DIR:  Path = ANALYTICS_DIR / "weekly_executive_review"
WEEKLY_EXECUTIVE_REVIEW_FILE: Path = WEEKLY_EXECUTIVE_REVIEW_DIR / "weekly_executive_review.json"

# ── Capital Deployment OS (Phase 5B) ─────────────────────────────────────────
CAPITAL_DEPLOYMENT_DIR:             Path = ANALYTICS_DIR / "capital_deployment"
CAPITAL_DEPLOYMENT_DECISIONS_FILE:  Path = CAPITAL_DEPLOYMENT_DIR / "deployment_decisions.jsonl"
CAPITAL_DEPLOYMENT_TELEMETRY_FILE:  Path = CAPITAL_DEPLOYMENT_DIR / "deployment_telemetry.jsonl"

# ── Exit convexity analytics layer ───────────────────────────────────────────
EXIT_ANALYTICS_DIR:          Path = RUNTIME_DIR / "exit_analytics"
EXIT_TRADES_DIR:             Path = EXIT_ANALYTICS_DIR / "trades"
EXIT_REPORTS_DIR:            Path = EXIT_ANALYTICS_DIR / "reports"
EXIT_HOLDING_PATHS_DIR:      Path = EXIT_ANALYTICS_DIR / "holding_paths"
EXIT_POLICY_REPLAYS_DIR:     Path = EXIT_ANALYTICS_DIR / "policy_replays"
EXIT_RECORDS_FILE:           Path = EXIT_TRADES_DIR / "exit_records.jsonl"
EXIT_POLICY_REPLAY_FILE:     Path = EXIT_POLICY_REPLAYS_DIR / "policy_replay_results.jsonl"

# ── Analytics policy bridge layer ────────────────────────────────────────────
POLICY_DIR:                  Path = RUNTIME_DIR / "policy"
POLICY_DECISIONS_FILE:       Path = POLICY_DIR / "policy_decisions.jsonl"
EXIT_POLICY_DECISIONS_FILE:  Path = POLICY_DIR / "exit_policy_decisions.jsonl"
PI_DECISIONS_FILE:           Path = POLICY_DIR / "portfolio_intelligence_decisions.jsonl"
EVS_ANALYTICS_FILE:          Path = ANALYTICS_DIR / "executed_vs_skipped.jsonl"

# ── Exit Intelligence OS ───────────────────────────────────────────────────────
EXIT_INTELLIGENCE_DIR:             Path = RUNTIME_DIR / "analytics" / "exit_intelligence"
EXIT_INTEL_OBSERVATIONS_FILE:      Path = EXIT_INTELLIGENCE_DIR / "exit_observations.jsonl"
EXIT_INTEL_PATH_SNAPSHOTS_FILE:    Path = EXIT_INTELLIGENCE_DIR / "exit_path_snapshots.jsonl"
EXIT_INTEL_EXEC_OBSERVATIONS_FILE: Path = EXIT_INTELLIGENCE_DIR / "execution_exit_observations.jsonl"
EXIT_INTEL_PORTFOLIO_PRESSURE_FILE:Path = EXIT_INTELLIGENCE_DIR / "portfolio_exit_pressure.jsonl"
EXIT_INTEL_VELOCITY_SCORES_FILE:   Path = EXIT_INTELLIGENCE_DIR / "exit_velocity_scores.jsonl"
EXIT_INTEL_LIFECYCLE_FILE:         Path = EXIT_INTELLIGENCE_DIR / "capital_lifecycle.jsonl"
EXIT_INTEL_DATASET_FILE:           Path = EXIT_INTELLIGENCE_DIR / "research_dataset.jsonl"
EXIT_INTEL_REPORTS_DIR:            Path = RUNTIME_DIR / "reports" / "exit_intelligence"
EXIT_INTEL_DAILY_REPORT_FILE:      Path = EXIT_INTEL_REPORTS_DIR / "exit_intelligence_daily_{date}.md"

# ── Compression continuation analytics ────────────────────────────────────────
COMPRESSION_CONTINUATION_DIR:        Path = ANALYTICS_DIR / "compression_continuation"
COMPRESSION_CONTINUATION_LOG_FILE:   Path = COMPRESSION_CONTINUATION_DIR / "compression_snapshots.jsonl"

# ── Suppression outcome attribution telemetry ─────────────────────────────────
SUPPRESSION_OUTCOME_DIR:             Path = ANALYTICS_DIR / "suppression_outcome"
SUPPRESSION_ACTIVE_STATE_FILE:       Path = SUPPRESSION_OUTCOME_DIR / "suppression_active_state.json"
SUPPRESSION_OUTCOME_LOG_FILE:        Path = SUPPRESSION_OUTCOME_DIR / "suppression_outcomes.jsonl"
SUPPRESSION_PHASE_TRANSITION_FILE:   Path = SUPPRESSION_OUTCOME_DIR / "suppression_phase_transitions.jsonl"

# ── Addon continuation telemetry ──────────────────────────────────────────────
ADDON_CONTINUATION_DIR:              Path = ANALYTICS_DIR / "addon_continuation"
ADDON_CONTINUATION_LOG_FILE:         Path = ADDON_CONTINUATION_DIR / "addon_continuation.jsonl"

# ── Holding expectancy telemetry ──────────────────────────────────────────────
HOLDING_EXPECTANCY_DIR:               Path = LOGS_DIR / "holding_expectancy"
HOLDING_EXPECTANCY_SNAPSHOT_FILE:     Path = HOLDING_EXPECTANCY_DIR / "holding_expectancy.jsonl"
HOLDING_EXPECTANCY_MATERIALIZED_FILE: Path = HOLDING_EXPECTANCY_DIR / "holding_expectancy_materialized.jsonl"

# ── Predictive expansion layer ────────────────────────────────────────────────
PREDICTIVE_EXPANSION_DIR:    Path = ANALYTICS_DIR / "predictive_expansion"
PREDICTIVE_SCORES_FILE:      Path = PREDICTIVE_EXPANSION_DIR / "predictive_scores.jsonl"
PREDICTIVE_REPORTS_DIR:      Path = PREDICTIVE_EXPANSION_DIR / "reports"

# ── Intraday expansion + shadow observation layer ────────────────────────────
INTRADAY_DIR:                 Path = RUNTIME_DIR / "intraday"
INTRADAY_EXPANSION_LOG_FILE:  Path = INTRADAY_DIR / "expansion_states.jsonl"
SHADOW_ENTRY_LOG_FILE:        Path = INTRADAY_DIR / "shadow_entries.jsonl"
SHADOW_STATS_FILE:            Path = INTRADAY_DIR / "shadow_stats.json"
PREDICTIVE_CANDIDATE_LOG_FILE: Path = INTRADAY_DIR / "candidate_rankings.jsonl"

# ── Future Leader shadow observer layer ───────────────────────────────────────
# observation_only — no execution connection
FUTURE_LEADER_DIR:             Path = RUNTIME_DIR / "future_leader"
FUTURE_LEADER_CANDIDATES_FILE: Path = FUTURE_LEADER_DIR / "candidates.jsonl"
FUTURE_LEADER_REPORTS_DIR:     Path = FUTURE_LEADER_DIR / "reports"
# Research integrity state — logs/future_leader_integrity/future_leader_integrity_YYYYMMDD.jsonl
FUTURE_LEADER_INTEGRITY_DIR:     Path = LOGS_DIR / "future_leader_integrity"
# Forward return telemetry — logs/future_leader_survivability/future_leader_survivability_YYYYMMDD.jsonl
FUTURE_LEADER_SURVIVABILITY_DIR: Path = LOGS_DIR / "future_leader_survivability"
# Failure transition telemetry — logs/future_leader_failure/future_leader_failure_YYYYMMDD.jsonl
FUTURE_LEADER_FAILURE_DIR: Path = LOGS_DIR / "future_leader_failure"
# Regime segmentation tags — logs/future_leader_regime/future_leader_regime_YYYYMMDD.jsonl
FUTURE_LEADER_REGIME_DIR: Path = LOGS_DIR / "future_leader_regime"
# Failure archetype collapse structure — logs/future_leader_archetype/future_leader_archetype_YYYYMMDD.jsonl
FUTURE_LEADER_ARCHETYPE_DIR: Path = LOGS_DIR / "future_leader_archetype"
# Failure transition sequence telemetry — logs/future_leader_transition/future_leader_transition_YYYYMMDD.jsonl
FUTURE_LEADER_TRANSITION_DIR: Path = LOGS_DIR / "future_leader_transition"

# ── Phase 4B: Micro Live Governance Bridge ────────────────────────────────────
GOVERNANCE_AUDIT_DIR:         Path = RUNTIME_DIR / "deployment" / "governance_audit"
GOVERNANCE_FREEZE_STATE_FILE: Path = GOVERNANCE_AUDIT_DIR / "freeze_state.json"
GOVERNANCE_REPLAY_HASH_FILE:  Path = GOVERNANCE_AUDIT_DIR / "governance_replay_hash.json"
# Missed alpha telemetry (analytics only — no execution mutation)
MISSED_ALPHA_LOG_DIR:         Path = LOGS_DIR / "missed_alpha"
MISSED_ALPHA_ENRICHED_DIR:    Path = LOGS_DIR / "missed_alpha_enriched"
# Early promotion telemetry (observation only — no auto-promotion)
EARLY_PROMOTION_LOG_DIR:      Path = LOGS_DIR / "early_promotion_telemetry"

# ── Phase 2.5: Universe Outcome Tracking (promotion / rotation / exploration) ─
UNIVERSE_OUTCOME_DIR:                  Path = LOGS_DIR / "universe"
PROMOTION_OUTCOME_FILE:                Path = UNIVERSE_OUTCOME_DIR / "promotion_outcome.jsonl"
ROTATION_OUTCOME_FILE:                 Path = UNIVERSE_OUTCOME_DIR / "rotation_outcome.jsonl"
EXPLORATION_OUTCOME_FILE:              Path = UNIVERSE_OUTCOME_DIR / "exploration_outcome.jsonl"
PROMOTION_OUTCOME_MATERIALIZED_FILE:   Path = UNIVERSE_OUTCOME_DIR / "promotion_outcome_materialized.jsonl"
ROTATION_OUTCOME_MATERIALIZED_FILE:    Path = UNIVERSE_OUTCOME_DIR / "rotation_outcome_materialized.jsonl"

# ── Phase 4B.1: Governance Diff + Human Confirmation Telemetry ───────────────
GOVERNANCE_DIFF_DIR:        Path = RUNTIME_DIR / "deployment" / "governance_diff"
HUMAN_CONFIRMATION_DIR:     Path = RUNTIME_DIR / "deployment" / "human_confirmation"

# ── Phase 5A: Future Leader Runtime Orchestrator ──────────────────────────────
RUNTIME_ORCHESTRATOR_DIR:          Path = RUNTIME_DIR / "runtime_orchestrator"
RUNTIME_ORCHESTRATOR_CHECKPOINT_DIR: Path = RUNTIME_ORCHESTRATOR_DIR / "checkpoints"
RUNTIME_ORCHESTRATOR_EVENTS_DIR:   Path = LOGS_DIR / "runtime" / "runtime_orchestrator"

# ── Winner add-on execution layer ─────────────────────────────────────────────
ADDON_DIR:               Path = RUNTIME_DIR / "addon"
ADDON_STATE_FILE:        Path = ADDON_DIR / "addon_state.json"
ADDON_DECISIONS_FILE:    Path = ADDON_DIR / "addon_decisions.jsonl"

# ── Research candidate state files (Study40/41/45 production integration) ────
EQ_SCALE_ADDON_STATE_FILE: Path = RUNTIME_DIR / "eq_scale_addon_state.json"
ATR_EXT_DEFER_STATE_FILE:  Path = RUNTIME_DIR / "atr_ext_defer_state.json"
VOL_ADJ_STATE_FILE:        Path = RUNTIME_DIR / "vol_adj_state.json"

# ── Quality Replacement Engine — Study57/58A ADOPT (Shadow mode) ──────────────
QUAL_REPLACE_AUDIT_FILE:    Path = LOGS_DIR    / "quality_replacement_audit.csv"
QUAL_REPLACE_MISSED_FILE:   Path = LOGS_DIR    / "quality_replacement_missed.csv"
QUAL_REPLACE_OUTCOMES_FILE: Path = LOGS_DIR    / "replacement_outcomes.csv"
QUAL_REPLACE_STATE_FILE:    Path = RUNTIME_DIR / "quality_replacement_state.json"

# ── Phase9 Shadow Audit (Study59) — observation only ─────────────────────────
QUAL_REPLACE_P9A_FILE:  Path = LOGS_DIR / "qr_phase9a_trigger_dist.csv"
QUAL_REPLACE_P9B_FILE:  Path = LOGS_DIR / "qr_phase9b_forward_attr.csv"
QUAL_REPLACE_P9C_FILE:  Path = LOGS_DIR / "qr_phase9c_missed_opp.csv"
QUAL_REPLACE_P9D_FILE:  Path = LOGS_DIR / "qr_phase9d_false_trigger.csv"
QUAL_REPLACE_P9E_DIR:   Path = LOGS_DIR / "qr_shadow_summary"

# ── Repository governance layer ───────────────────────────────────────────────
GOVERNANCE_DIR:          Path = BASE_DIR / "state"
STRATEGY_REGISTRY_FILE:  Path = GOVERNANCE_DIR / "strategy_registry.json"
REPO_INVENTORY_FILE:     Path = GOVERNANCE_DIR / "repo_inventory.json"
TOOLS_DIR:               Path = BASE_DIR / "tools"

ARTIFACTS_DIR:           Path = BASE_DIR / "artifacts"
FEATURE_CACHE_DIR:       Path = ARTIFACTS_DIR / "features"
FEATURE_SNAPSHOT_DIR:    Path = FEATURE_CACHE_DIR / "snapshots"


# ────────────────────────────────────────────────────────────────────────────
# ライブ運用安全設定
# ────────────────────────────────────────────────────────────────────────────
LIVE_MODE: bool = _env_str("LIVE_MODE", "false").lower() == "true"
MAX_ORDERS_PER_DAY: int = _env_int("MAX_ORDERS_PER_DAY", 20, minimum=1)
MAX_SINGLE_POSITION_YEN: int = _env_int("MAX_SINGLE_POSITION_YEN", 600000, minimum=1)
ORDER_RATE_LIMIT_SEC: float = _env_float("ORDER_RATE_LIMIT_SEC", 5.0, minimum=0.0)
ALLOW_YFINANCE_NETWORK: bool = _env_str("ALLOW_YFINANCE_NETWORK", "false").lower() == "true"
DEFAULT_DATA_VERSION: str = _env_str("DATA_VERSION", "").strip() or latest_dataset_version()

# ────────────────────────────────────────────────────────────────────────────
# kabuステーション API 接続設定
# ────────────────────────────────────────────────────────────────────────────
KABUS_HOST:     str = _env_str("KABUS_HOST", "localhost")
KABUS_PORT:     int = _env_int("KABUS_PORT", 18080, minimum=1)
KABUS_PASSWORD: str = _env_str("KABU_API_PASSWORD", "")

# 発注タイムスタンプ記録ファイル（レートリミット用）
_LAST_ORDER_FILE: Path = RUNTIME_DIR / "_last_order_ts.json"


# ── J-Quants execution infrastructure (Study75 prerequisite) ────────────────
JQUANTS_DATA_DIR:      Path = DATASET_DIR / "jquants"
JQUANTS_RAW_DIR:       Path = JQUANTS_DATA_DIR / "raw"
JQUANTS_CACHE_DIR:     Path = JQUANTS_DATA_DIR / "cache"
JQUANTS_STAGING_DIR:   Path = JQUANTS_CACHE_DIR / "staging"
JQUANTS_DAILY_STAGING_DIR: Path = JQUANTS_CACHE_DIR / "daily"  # Strategy C（日次イテレーション・study75_downloader.py）
JQUANTS_PROCESSED_DIR: Path = JQUANTS_DATA_DIR / "processed"
JQUANTS_METADATA_DIR:  Path = JQUANTS_DATA_DIR / "metadata"
JQUANTS_LOGS_DIR:      Path = JQUANTS_DATA_DIR / "logs"

# J-Quants API v2（2026-07時点の公式クイックスタート準拠）: 認証は x-api-key ヘッダーの静的APIキー。
# ダッシュボード [設定 » APIキー] で発行。mailaddress/password や refreshToken/idToken のフローは使わない。
JQUANTS_API_KEY: str = _env_str("JQUANTS_API_KEY", "")


def ensure_jquants_dirs() -> None:
    """data/jquants/ 配下の標準ディレクトリ群を事前作成する。"""
    for path in (
        JQUANTS_RAW_DIR, JQUANTS_CACHE_DIR, JQUANTS_STAGING_DIR, JQUANTS_DAILY_STAGING_DIR,
        JQUANTS_PROCESSED_DIR, JQUANTS_METADATA_DIR, JQUANTS_LOGS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


# ── 日本株分析データベース（database/market・Single Source of Truth） ────────
# data/（バックテスト生成物）・cache/（売買システム専用）・data/jquants/（取り込みLegacy）とは
# 完全独立。将来のセクター/ETF/RS/ファクター/ML分析はこの配下のみを参照する（src/database/参照）。
DATABASE_MARKET_DIR:       Path = BASE_DIR / "database" / "market"
DATABASE_OHLCV_DIR:        Path = DATABASE_MARKET_DIR / "ohlcv"
DATABASE_MASTER_DIR:       Path = DATABASE_MARKET_DIR / "master"
DATABASE_FUNDAMENTALS_DIR: Path = DATABASE_MARKET_DIR / "fundamentals"
DATABASE_ETF_DIR:          Path = DATABASE_MARKET_DIR / "etf"
DATABASE_INDEX_DIR:        Path = DATABASE_MARKET_DIR / "index"
DATABASE_FACTOR_DIR:       Path = DATABASE_MARKET_DIR / "factor"
DATABASE_MACRO_DIR:        Path = DATABASE_MARKET_DIR / "macro"
DATABASE_MARGIN_DIR:       Path = DATABASE_MARKET_DIR / "margin"
DATABASE_SHORTSELLING_DIR: Path = DATABASE_MARKET_DIR / "shortselling"
DATABASE_METADATA_DIR:     Path = DATABASE_MARKET_DIR / "metadata"
DATABASE_CACHE_DIR:        Path = DATABASE_MARKET_DIR / "cache"
DATABASE_MINUTE_DIR:       Path = DATABASE_MARKET_DIR / "minute"
DATABASE_TICK_DIR:         Path = DATABASE_MARKET_DIR / "tick"

# ── Bulk API 原本CSV.GZ 永久保存領域（database/market・dataとは独立・絶対コミット禁止） ──
# J-Quants Bulk API（/v2/bulk/list, /v2/bulk/get）のKeyパスをそのままミラーする。
# Parquet（database/market/配下）はこの原本からの派生データという位置付け（解約後もこれが一次データ）。
ARCHIVE_DIR:      Path = BASE_DIR / "archive"
ARCHIVE_BULK_DIR: Path = ARCHIVE_DIR / "bulk"


def ensure_database_market_dirs() -> None:
    """database/market/ 配下の標準ディレクトリ群を事前作成する。"""
    for path in (
        DATABASE_OHLCV_DIR, DATABASE_MASTER_DIR, DATABASE_FUNDAMENTALS_DIR, DATABASE_ETF_DIR,
        DATABASE_INDEX_DIR, DATABASE_FACTOR_DIR, DATABASE_MACRO_DIR, DATABASE_MARGIN_DIR,
        DATABASE_SHORTSELLING_DIR, DATABASE_METADATA_DIR, DATABASE_CACHE_DIR,
        DATABASE_MINUTE_DIR, DATABASE_TICK_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def ensure_archive_dirs() -> None:
    """archive/bulk/ 配下（Bulk API原本CSV.GZ永久保存領域）を事前作成する。"""
    ARCHIVE_BULK_DIR.mkdir(parents=True, exist_ok=True)


# ── Study75/76 research pipeline ─────────────────────────────────────────────
# Study75 完了後に生成される規則ユニバース（月次再適用・survivorship-free）ファイル。
# スキーマ: {"monthly_universe": {"YYYY-MM-DD": ["1301.T", ...], ...}, "meta": {...}}
#           key は各月の再適用日（第1営業日）。Study76/77 共通で使用（正典: study76_execution_plan.md §2.2）。
# 未生成の間は STUDY75_UNIVERSE_FILE 環境変数か --universe-file 引数で明示指定する。
STUDY75_UNIVERSE_FILE: Path = _resolve_path(
    "STUDY75_UNIVERSE_FILE", UNIVERSE_DIR / "study75_survivorship_free.json"
)


def ensure_base_dirs() -> None:
    """paths.py が管理する標準ディレクトリ群を事前作成する。"""
    for path in (
        RESULTS_DIR,
        LOGS_DIR,
        CACHE_DIR,
        DATASET_DIR,
        RUNTIME_DIR,
        REPORTS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def assert_live_ready() -> None:
    """
    ライブ発注の直前に呼ぶ安全チェック（4項目一括検証）。
    条件を満たさない場合は RuntimeError を送出してプロセスを止める。
    """
    errors: list[str] = []

    if not LIVE_MODE:
        errors.append(
            "LIVE_MODE が有効ではありません。\n"
            "  .env に LIVE_MODE=true を追加してください。"
        )

    if not LIVE_UNIVERSE_FILE.exists():
        errors.append(
            f"ユニバースファイルが見つかりません: {LIVE_UNIVERSE_FILE}"
        )

    if MAX_ORDERS_PER_DAY <= 0:
        errors.append(
            f"MAX_ORDERS_PER_DAY={MAX_ORDERS_PER_DAY} が不正です（1以上を設定してください）。"
        )

    if errors:
        raise RuntimeError(
            "ライブ発注の安全チェックに失敗しました:\n" +
            "\n".join(f"  [{i+1}] {e}" for i, e in enumerate(errors))
        )

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)


# ────────────────────────────────────────────────────────────────────────────
# 実行コンテキスト検証（LIVE_MODE 時のスクリプト許可リスト）
# ────────────────────────────────────────────────────────────────────────────
_LIVE_ALLOWED_SCRIPTS: frozenset[str] = frozenset({
    "run_live_signal.py",
    "run_morning_signal.py",
})


def assert_execution_context() -> None:
    """
    呼び出し元スクリプトが実発注を許可されているか検証する。
    LIVE_MODE=true のとき、許可リスト外から呼ばれた場合は RuntimeError を送出。
    """
    if not LIVE_MODE:
        return

    import sys
    script = Path(sys.argv[0]).resolve().name

    if script not in _LIVE_ALLOWED_SCRIPTS:
        raise RuntimeError(
            f"実発注ブロック: '{script}' は LIVE_MODE での実行が許可されていません。\n"
            f"  許可スクリプト: {sorted(_LIVE_ALLOWED_SCRIPTS)}\n"
            "  ドライランで実行するか、許可リスト(_LIVE_ALLOWED_SCRIPTS)を確認してください。"
        )


# ────────────────────────────────────────────────────────────────────────────
# データ整合性チェック
# ────────────────────────────────────────────────────────────────────────────
def assert_kabus_connection() -> None:
    """
    kabuステーション API 接続設定の最低限チェック。
    接続前に必ず呼ぶ。
    """
    errors: list[str] = []

    if not KABUS_HOST:
        errors.append("KABUS_HOST が未設定です")

    if not isinstance(KABUS_PORT, int) or KABUS_PORT <= 0:
        errors.append(f"KABUS_PORT={KABUS_PORT} が不正です")

    if not KABUS_PASSWORD:
        errors.append(
            "KABU_API_PASSWORD が未設定です。\n"
            "  .env に KABU_API_PASSWORD=<パスワード> を設定してください。"
        )

    if errors:
        raise RuntimeError(
            "kabuステーション接続設定エラー:\n" +
            "\n".join(f"  [{i+1}] {e}" for i, e in enumerate(errors))
        )


def acquire_runtime_lock(mode: str = "live", run_id: str = ""):
    """
    二重起動防止ロックを取得する（liveness-aware 版）。
    プロセス終了時に自動解放（atexit 登録）。

    旧実装（ファイル存在チェックのみ）を ExecutionLock に委譲。
    陳腐化ロック（PID消滅・heartbeat失効 30s）は自動回収する。
    生きているプロセスが保持している場合のみ RuntimeError を送出。

    Returns:
        ExecutionLock instance (use to start HeartbeatThread).
    """
    import atexit
    from src.runtime.execution_lock import ExecutionLock, ExecutionLockError

    global _RUNTIME_LOCK_INSTANCE
    try:
        lock = ExecutionLock(
            EXECUTION_LOCK_FILE,
            mode=mode,
            run_id=run_id,
            heartbeat_expiry_sec=30.0,   # stale reclaim threshold
        )
        lock.acquire()
        _RUNTIME_LOCK_INSTANCE = lock
        atexit.register(release_runtime_lock)
        return lock
    except ExecutionLockError as exc:
        data = exc.lock_data
        raise RuntimeError(
            f"別のインスタンスが実行中です (PID={data.get('pid', '?')})。\n"
            f"  {EXECUTION_LOCK_FILE} を確認してください。\n"
            "  プロセスが終了していれば次回起動時に自動回収されます。"
        ) from exc


_RUNTIME_LOCK_INSTANCE = None


def release_runtime_lock() -> None:
    """ロックファイルを解放する（ExecutionLock 経由）。"""
    global _RUNTIME_LOCK_INSTANCE
    if _RUNTIME_LOCK_INSTANCE is not None:
        try:
            _RUNTIME_LOCK_INSTANCE.release()
        except Exception:
            pass
        _RUNTIME_LOCK_INSTANCE = None
    else:
        # fallback for old callers
        try:
            EXECUTION_LOCK_FILE.unlink(missing_ok=True)
        except Exception:
            pass


def enforce_order_rate_limit() -> None:
    """
    直前の発注から ORDER_RATE_LIMIT_SEC 秒未満の場合は RuntimeError を送出する。
    kabuステーション API の過剰発注 BAN 防止。
    """
    if not _LAST_ORDER_FILE.exists():
        return
    try:
        data = json.loads(_LAST_ORDER_FILE.read_text(encoding="utf-8"))
        elapsed = time.time() - data.get("ts", 0.0)
        if elapsed < ORDER_RATE_LIMIT_SEC:
            wait = ORDER_RATE_LIMIT_SEC - elapsed
            raise RuntimeError(
                f"レートリミット: あと {wait:.1f}秒 待機が必要です "
                f"（ORDER_RATE_LIMIT_SEC={ORDER_RATE_LIMIT_SEC}秒）。"
            )
    except (json.JSONDecodeError, KeyError):
        pass


def record_order_sent() -> None:
    """発注タイムスタンプを記録する（enforce_order_rate_limit 用）。"""
    _LAST_ORDER_FILE.parent.mkdir(parents=True, exist_ok=True)
    _LAST_ORDER_FILE.write_text(
        json.dumps({"ts": time.time()}),
        encoding="utf-8",
    )


def verify_dataset_integrity(data_version: str = "") -> None:
    """
    バックテスト用データセットの存在を確認する。

    Args:
        data_version: 確認する DATA_VERSION（例 "2026-03-28"）。
                      空文字の場合は BACKTEST_DATASET_DIR の存在のみ確認。

    Raises:
        RuntimeError: 必要なパスが存在しない場合。
    """
    errors: list[str] = []

    if not BACKTEST_DATASET_DIR.exists():
        errors.append(
            f"backtest_dataset ディレクトリが存在しません: {BACKTEST_DATASET_DIR}\n"
            "  先に build_dataset_snapshot.py を実行してください。"
        )
    elif data_version:
        snapshot_dir = BACKTEST_DATASET_DIR / data_version
        if not snapshot_dir.exists():
            errors.append(
                f"スナップショット '{data_version}' が存在しません: {snapshot_dir}\n"
                f"  DATA_VERSION={data_version} python build_dataset_snapshot.py"
            )
        else:
            meta = snapshot_dir / "_meta.json"
            parquets = list(snapshot_dir.glob("*.parquet"))
            if not meta.exists():
                errors.append(f"_meta.json が見つかりません: {meta}")
            if not parquets:
                errors.append(f"parquet ファイルが見つかりません: {snapshot_dir}")

    if not RSR_UNIVERSE_FILE.exists():
        errors.append(f"RSRユニバース CSV が見つかりません: {RSR_UNIVERSE_FILE}")

    if errors:
        raise RuntimeError(
            "データ整合性チェックに失敗しました:\n" +
            "\n".join(f"  [{i+1}] {e}" for i, e in enumerate(errors))
        )
