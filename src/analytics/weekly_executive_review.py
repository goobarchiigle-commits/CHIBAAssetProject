"""
weekly_executive_review.py — Executive Weekly Review (Phase 7B)

Single weekly report summarising the entire self-improving trading OS.
Observation only.  Fail-open.  No strategy / allocation / governance changes.

Inputs (all read-only, 7-day lookback window):
  EVIDENCE_PROMOTION_FILE
  SHADOW_RECOMMENDATION_FILE
  PROMOTION_READINESS_FILE
  PRODUCTION_PROMOTION_GATE_FILE
  ACTIVE_EXPERIMENT_REGISTRY_FILE
  CAPITAL_ESCALATION_FILE
  ROLLBACK_GOVERNANCE_FILE
  IMPROVEMENT_RANKING_FILE
  SYSTEM_HEALTH_REPORT_FILE
  EXPERIMENT_PERFORMANCE_FILE

Output:
  weekly_executive_review.json — overwritten each run (point-in-time snapshot)

Sections:
  1. system_health       — HEALTHY/WARNING/CRITICAL, stale/missing/orphaned
  2. experiment_summary  — active, escalated, rollback-candidate, rolled-back
  3. improvement_ranking — top-10 and bottom-10
  4. capital_deployment  — allocation by experiment; week changes
  5. rollback_activity   — experiments stopped; trigger reasons
  6. performance_summary — aggregate PnL, avg expectancy, avg drawdown
  7. key_alerts          — severity-ranked items requiring human review

Design:
  observation_only  — no writes except the review JSON itself
  fail-open         — each section wrapped independently; partial output on error
  deterministic     — same inputs → same JSON
  weekly_window     — records older than LOOKBACK_DAYS are excluded from sections
                      that measure "this week"; registry / ranking are always latest
"""
from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

_JST = timezone(timedelta(hours=9))
SCHEMA_VERSION = "v1"

LOOKBACK_DAYS = 7   # "this week" window

# Alert severities
ALERT_CRITICAL = "CRITICAL"
ALERT_WARNING  = "WARNING"
ALERT_INFO     = "INFO"

# Allocation ladder (mirrors progressive_capital_escalation)
_ALLOC_LEVEL_PCT: Dict[int, float] = {
    0: 0.005, 1: 0.010, 2: 0.020,
    3: 0.050, 4: 0.100, 5: 0.250,
}

_ESC_DECISIONS = {
    "ESCALATE_LEVEL_1", "ESCALATE_LEVEL_2",
    "ESCALATE_LEVEL_3", "FULL_SCALE",
}


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class SystemHealthSection:
    overall_status:       str
    stale_telemetry:      List[str]
    missing_telemetry:    List[str]
    orphaned_experiments: List[str]
    check_count:          int
    warning_count:        int
    critical_count:       int


@dataclass
class ExperimentEntry:
    experiment_id:    str
    status:           str
    allocation_level: int
    allocation_pct:   float
    start_date:       str


@dataclass
class ExperimentSummarySection:
    active_experiments:      List[ExperimentEntry]
    escalated_this_week:     List[str]
    rollback_candidates:     List[str]
    rolled_back_this_week:   List[str]
    total_active:            int


@dataclass
class RankedExperiment:
    rank:          int
    experiment_id: str
    ranking_score: float
    total_pnl:     float
    win_rate:      float
    max_drawdown:  float


@dataclass
class ImprovementRankingSection:
    top_10:    List[RankedExperiment]
    bottom_10: List[RankedExperiment]
    total_ranked: int


@dataclass
class AllocationChange:
    experiment_id: str
    date:          str
    decision:      str
    previous_level: int
    new_level:     int
    allocation_pct: float


@dataclass
class CapitalDeploymentSection:
    allocation_by_experiment: Dict[str, float]   # eid → allocation_pct
    changes_this_week:        List[AllocationChange]
    total_allocated_pct:      float              # sum across experiments


@dataclass
class RollbackEvent:
    experiment_id:   str
    date:            str
    decision:        str
    trigger_reasons: List[str]


@dataclass
class RollbackActivitySection:
    events_this_week:       List[RollbackEvent]
    experiments_stopped:    List[str]            # ROLLED_BACK this week
    rollback_candidate_ids: List[str]            # ROLLBACK_CANDIDATE this week


@dataclass
class PerformanceSummarySection:
    aggregate_pnl:   float
    avg_expectancy:  float
    avg_drawdown:    float
    avg_win_rate:    float
    total_trades:    int
    experiment_count: int   # experiments contributing to the summary


@dataclass
class Alert:
    severity:      str     # ALERT_CRITICAL / ALERT_WARNING / ALERT_INFO
    message:       str
    experiment_id: str = ""
    source:        str = ""


@dataclass
class KeyAlertsSection:
    alerts: List[Alert]
    critical_count: int
    warning_count:  int
    info_count:     int


@dataclass
class WeeklyExecutiveReview:
    review_date:      str
    lookback_days:    int
    system_health:    SystemHealthSection
    experiment_summary: ExperimentSummarySection
    improvement_ranking: ImprovementRankingSection
    capital_deployment:  CapitalDeploymentSection
    rollback_activity:   RollbackActivitySection
    performance_summary: PerformanceSummarySection
    key_alerts:          KeyAlertsSection
    schema_version:      str = SCHEMA_VERSION


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_float(v: Any, fallback: float = 0.0) -> float:
    if v is None:
        return fallback
    try:
        f = float(v)
        return f if math.isfinite(f) else fallback
    except (TypeError, ValueError):
        return fallback


def _safe_int(v: Any, fallback: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return fallback


def _parse_date(s: str) -> Optional[datetime]:
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s[:len(fmt)], fmt)
        except Exception:
            pass
    try:
        return datetime.fromisoformat(s[:19])
    except Exception:
        return None


def _load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    try:
        if not path.exists():
            return rows
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    except Exception as exc:
        logger.debug("[WER] load_jsonl %s: %s", path, exc)
    return rows


def _load_json(path: Path) -> Optional[dict]:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.debug("[WER] load_json %s: %s", path, exc)
        return None


def _load_registry(path: Path) -> List[dict]:
    try:
        raw = _load_json(path)
        if raw is None:
            return []
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            exps = raw.get("experiments")
            if isinstance(exps, list):
                return exps
        return []
    except Exception as exc:
        logger.debug("[WER] load_registry %s: %s", path, exc)
        return []


def _within_window(date_str: str, today: datetime, days: int) -> bool:
    d = _parse_date(date_str)
    if d is None:
        return False
    d = d.replace(tzinfo=None)
    cutoff = today.replace(tzinfo=None) - timedelta(days=days)
    return d >= cutoff


def _today_naive(today: str) -> datetime:
    d = _parse_date(today)
    return (d if d else datetime.now(_JST)).replace(tzinfo=None)


def _alloc_pct(level: int) -> float:
    return _ALLOC_LEVEL_PCT.get(max(0, min(5, level)), 0.005)


# ── Section builders ──────────────────────────────────────────────────────────

def _build_system_health(health_report: Optional[dict]) -> SystemHealthSection:
    """Extract system health info from the health audit JSON snapshot."""
    try:
        if health_report is None:
            return SystemHealthSection(
                overall_status="UNKNOWN",
                stale_telemetry=[],
                missing_telemetry=[],
                orphaned_experiments=[],
                check_count=0,
                warning_count=0,
                critical_count=0,
            )

        checks = health_report.get("checks", [])
        stale: List[str] = []
        missing: List[str] = []
        orphaned: List[str] = []
        warn_count = 0
        crit_count = 0

        for c in checks:
            status = str(c.get("status", ""))
            name   = str(c.get("check_name", ""))
            msg    = str(c.get("message", "")).lower()
            if status == "WARNING":
                warn_count += 1
            elif status == "CRITICAL":
                crit_count += 1

            if "stale" in msg:
                stale.append(name)
            if "not found" in msg or "no valid records" in msg:
                missing.append(name)

            if name == "orphaned_registry_entries" and status != "HEALTHY":
                details = c.get("details", {})
                orphaned = list(details.get("orphans", []))

        summary = health_report.get("summary", {})
        # Also pick up orphans from summary if present
        if not orphaned and summary:
            pass  # summary doesn't carry orphans directly; already captured above

        return SystemHealthSection(
            overall_status=str(health_report.get("overall_status", "UNKNOWN")),
            stale_telemetry=stale,
            missing_telemetry=missing,
            orphaned_experiments=orphaned,
            check_count=len(checks),
            warning_count=warn_count,
            critical_count=crit_count,
        )
    except Exception as exc:
        logger.debug("[WER] _build_system_health failed: %s", exc)
        return SystemHealthSection(
            overall_status="UNKNOWN", stale_telemetry=[], missing_telemetry=[],
            orphaned_experiments=[], check_count=0, warning_count=0, critical_count=0,
        )


def _build_experiment_summary(
    registry: List[dict],
    escalation_rows: List[dict],
    rollback_rows:   List[dict],
    today:           datetime,
) -> ExperimentSummarySection:
    try:
        entries: List[ExperimentEntry] = []
        for r in registry:
            eid    = str(r.get("experiment_id", ""))
            status = str(r.get("status", ""))
            if not eid or status != "ACTIVE":
                continue
            level = _safe_int(r.get("allocation_level"), 0)
            entries.append(ExperimentEntry(
                experiment_id=eid,
                status=status,
                allocation_level=level,
                allocation_pct=_alloc_pct(level),
                start_date=str(r.get("start_date", "")),
            ))

        escalated: List[str] = []
        for r in escalation_rows:
            if str(r.get("decision", "")) in _ESC_DECISIONS:
                date_f = str(r.get("evaluation_date", r.get("date", "")))
                if _within_window(date_f, today, LOOKBACK_DAYS):
                    eid = str(r.get("experiment_id", ""))
                    if eid and eid not in escalated:
                        escalated.append(eid)

        rb_candidates: List[str] = []
        rb_stopped:    List[str] = []
        for r in rollback_rows:
            decision = str(r.get("decision", ""))
            date_f   = str(r.get("evaluation_date", r.get("date", "")))
            eid      = str(r.get("experiment_id", ""))
            if not eid or not _within_window(date_f, today, LOOKBACK_DAYS):
                continue
            if decision == "ROLLBACK_CANDIDATE" and eid not in rb_candidates:
                rb_candidates.append(eid)
            if decision == "ROLLED_BACK" and eid not in rb_stopped:
                rb_stopped.append(eid)

        return ExperimentSummarySection(
            active_experiments=entries,
            escalated_this_week=escalated,
            rollback_candidates=rb_candidates,
            rolled_back_this_week=rb_stopped,
            total_active=len(entries),
        )
    except Exception as exc:
        logger.debug("[WER] _build_experiment_summary failed: %s", exc)
        return ExperimentSummarySection(
            active_experiments=[], escalated_this_week=[],
            rollback_candidates=[], rolled_back_this_week=[], total_active=0,
        )


def _build_improvement_ranking(ranking_rows: List[dict]) -> ImprovementRankingSection:
    try:
        # Latest record per experiment
        latest: Dict[str, dict] = {}
        for r in ranking_rows:
            eid  = str(r.get("experiment_id", ""))
            date = str(r.get("date", ""))
            if not eid:
                continue
            if eid not in latest or date > str(latest[eid].get("date", "")):
                latest[eid] = r

        ranked = sorted(latest.values(), key=lambda r: _safe_int(r.get("rank"), 9999))

        def _to_entry(r: dict) -> RankedExperiment:
            metrics = r.get("metrics", {})
            return RankedExperiment(
                rank=_safe_int(r.get("rank"), 0),
                experiment_id=str(r.get("experiment_id", "")),
                ranking_score=_safe_float(r.get("ranking_score"), 0.0),
                total_pnl=_safe_float(metrics.get("total_pnl"), 0.0),
                win_rate=_safe_float(metrics.get("win_rate"), 0.0),
                max_drawdown=_safe_float(metrics.get("max_drawdown"), 0.0),
            )

        top10    = [_to_entry(r) for r in ranked[:10]]
        bottom10 = [_to_entry(r) for r in ranked[-10:][::-1]]  # worst first

        return ImprovementRankingSection(
            top_10=top10,
            bottom_10=bottom10,
            total_ranked=len(ranked),
        )
    except Exception as exc:
        logger.debug("[WER] _build_improvement_ranking failed: %s", exc)
        return ImprovementRankingSection(top_10=[], bottom_10=[], total_ranked=0)


def _build_capital_deployment(
    registry:        List[dict],
    escalation_rows: List[dict],
    today:           datetime,
) -> CapitalDeploymentSection:
    try:
        # Current allocation by experiment (ACTIVE only, latest level)
        alloc: Dict[str, float] = {}
        for r in registry:
            eid    = str(r.get("experiment_id", ""))
            status = str(r.get("status", ""))
            if not eid or status != "ACTIVE":
                continue
            level = _safe_int(r.get("allocation_level"), 0)
            alloc[eid] = _alloc_pct(level)

        # Changes this week from escalation JSONL
        changes: List[AllocationChange] = []
        seen: Set[str] = set()
        for r in escalation_rows:
            decision = str(r.get("decision", ""))
            date_f   = str(r.get("evaluation_date", r.get("date", "")))
            eid      = str(r.get("experiment_id", ""))
            if not eid or not _within_window(date_f, today, LOOKBACK_DAYS):
                continue
            if decision in _ESC_DECISIONS or decision == "DEESCALATE":
                key = f"{eid}:{date_f}:{decision}"
                if key in seen:
                    continue
                seen.add(key)
                prev_level = _safe_int(r.get("previous_level"), 0)
                new_level  = _safe_int(r.get("new_level"), 0)
                changes.append(AllocationChange(
                    experiment_id=eid,
                    date=date_f,
                    decision=decision,
                    previous_level=prev_level,
                    new_level=new_level,
                    allocation_pct=_alloc_pct(new_level),
                ))

        total_pct = round(sum(alloc.values()), 4)

        return CapitalDeploymentSection(
            allocation_by_experiment=alloc,
            changes_this_week=changes,
            total_allocated_pct=total_pct,
        )
    except Exception as exc:
        logger.debug("[WER] _build_capital_deployment failed: %s", exc)
        return CapitalDeploymentSection(
            allocation_by_experiment={}, changes_this_week=[], total_allocated_pct=0.0,
        )


def _build_rollback_activity(
    rollback_rows: List[dict],
    today:         datetime,
) -> RollbackActivitySection:
    try:
        events: List[RollbackEvent] = []
        seen: Set[str] = set()
        for r in rollback_rows:
            decision = str(r.get("decision", ""))
            date_f   = str(r.get("evaluation_date", r.get("date", "")))
            eid      = str(r.get("experiment_id", ""))
            if not eid or not _within_window(date_f, today, LOOKBACK_DAYS):
                continue
            if decision not in ("ROLLED_BACK", "ROLLBACK_CANDIDATE", "REVIEW_REQUIRED"):
                continue
            key = f"{eid}:{date_f}:{decision}"
            if key in seen:
                continue
            seen.add(key)
            reasons = list(r.get("trigger_reasons", []))
            events.append(RollbackEvent(
                experiment_id=eid,
                date=date_f,
                decision=decision,
                trigger_reasons=reasons,
            ))

        stopped    = sorted({e.experiment_id for e in events if e.decision == "ROLLED_BACK"})
        candidates = sorted({e.experiment_id for e in events if e.decision == "ROLLBACK_CANDIDATE"})

        return RollbackActivitySection(
            events_this_week=events,
            experiments_stopped=stopped,
            rollback_candidate_ids=candidates,
        )
    except Exception as exc:
        logger.debug("[WER] _build_rollback_activity failed: %s", exc)
        return RollbackActivitySection(
            events_this_week=[], experiments_stopped=[], rollback_candidate_ids=[],
        )


def _build_performance_summary(
    perf_rows: List[dict],
    today:     datetime,
) -> PerformanceSummarySection:
    try:
        # Weekly EXPERIMENT group records
        rows = [
            r for r in perf_rows
            if str(r.get("group", "")) == "EXPERIMENT"
            and _within_window(str(r.get("date", "")), today, LOOKBACK_DAYS)
        ]
        if not rows:
            return PerformanceSummarySection(
                aggregate_pnl=0.0, avg_expectancy=0.0, avg_drawdown=0.0,
                avg_win_rate=0.0, total_trades=0, experiment_count=0,
            )

        total_pnl  = sum(_safe_float(r.get("pnl"), 0.0) for r in rows)
        total_trades = sum(_safe_int(r.get("trades"), 0) for r in rows)
        exp_vals   = [_safe_float(r.get("expectancy"), 0.0) for r in rows]
        dd_vals    = [_safe_float(r.get("max_drawdown"), 0.0) for r in rows]
        wr_vals    = [_safe_float(r.get("win_rate"), 0.0) for r in rows]

        avg_exp = sum(exp_vals) / len(exp_vals) if exp_vals else 0.0
        avg_dd  = sum(dd_vals)  / len(dd_vals)  if dd_vals  else 0.0
        avg_wr  = sum(wr_vals)  / len(wr_vals)  if wr_vals  else 0.0
        eids    = {str(r.get("experiment_id", "")) for r in rows if r.get("experiment_id")}

        return PerformanceSummarySection(
            aggregate_pnl=round(total_pnl, 2),
            avg_expectancy=round(avg_exp, 4),
            avg_drawdown=round(avg_dd, 4),
            avg_win_rate=round(avg_wr, 4),
            total_trades=total_trades,
            experiment_count=len(eids),
        )
    except Exception as exc:
        logger.debug("[WER] _build_performance_summary failed: %s", exc)
        return PerformanceSummarySection(
            aggregate_pnl=0.0, avg_expectancy=0.0, avg_drawdown=0.0,
            avg_win_rate=0.0, total_trades=0, experiment_count=0,
        )


def _build_key_alerts(
    health:     SystemHealthSection,
    exp_sum:    ExperimentSummarySection,
    rollback:   RollbackActivitySection,
    perf:       PerformanceSummarySection,
    ranking:    ImprovementRankingSection,
    deployment: CapitalDeploymentSection,
) -> KeyAlertsSection:
    alerts: List[Alert] = []

    try:
        # System health alerts
        if health.overall_status == "CRITICAL":
            alerts.append(Alert(
                severity=ALERT_CRITICAL,
                message="System health is CRITICAL — immediate review required",
                source="system_health",
            ))
        elif health.overall_status == "WARNING":
            alerts.append(Alert(
                severity=ALERT_WARNING,
                message=f"System health is WARNING ({health.warning_count} checks failed)",
                source="system_health",
            ))

        if health.missing_telemetry:
            alerts.append(Alert(
                severity=ALERT_WARNING,
                message=f"Missing telemetry in: {', '.join(health.missing_telemetry)}",
                source="system_health",
            ))

        if health.stale_telemetry:
            alerts.append(Alert(
                severity=ALERT_WARNING,
                message=f"Stale telemetry (>{3}d) in: {', '.join(health.stale_telemetry)}",
                source="system_health",
            ))

        if health.orphaned_experiments:
            alerts.append(Alert(
                severity=ALERT_WARNING,
                message=f"Orphaned registry entries: {', '.join(health.orphaned_experiments)}",
                source="system_health",
            ))

        # Rollback alerts
        for eid in rollback.experiments_stopped:
            alerts.append(Alert(
                severity=ALERT_CRITICAL,
                message=f"Experiment rolled back this week",
                experiment_id=eid,
                source="rollback_governance",
            ))

        for eid in rollback.rollback_candidate_ids:
            alerts.append(Alert(
                severity=ALERT_WARNING,
                message="Experiment is ROLLBACK_CANDIDATE — persisting for 3 days triggers auto-rollback",
                experiment_id=eid,
                source="rollback_governance",
            ))

        # Performance alerts
        if perf.aggregate_pnl < 0:
            alerts.append(Alert(
                severity=ALERT_WARNING,
                message=f"Weekly aggregate PnL is negative: {perf.aggregate_pnl:+.2f}",
                source="performance_summary",
            ))

        if perf.avg_drawdown < -0.10:
            alerts.append(Alert(
                severity=ALERT_WARNING,
                message=f"Average drawdown exceeds -10%: {perf.avg_drawdown:.1%}",
                source="performance_summary",
            ))

        # Capital concentration alert
        if deployment.total_allocated_pct > 0.50:
            alerts.append(Alert(
                severity=ALERT_WARNING,
                message=f"Total allocated capital is high: {deployment.total_allocated_pct:.1%}",
                source="capital_deployment",
            ))

        # No active experiments
        if exp_sum.total_active == 0:
            alerts.append(Alert(
                severity=ALERT_INFO,
                message="No active experiments — system is idle",
                source="experiment_summary",
            ))

        # Ranking gap alert: top vs bottom score spread
        if ranking.top_10 and ranking.bottom_10:
            top_score = ranking.top_10[0].ranking_score
            bot_score = ranking.bottom_10[0].ranking_score
            if top_score - bot_score > 60:
                alerts.append(Alert(
                    severity=ALERT_INFO,
                    message=(
                        f"Large ranking gap: top={top_score:.1f} vs "
                        f"bottom={bot_score:.1f} — consider retiring low-ranked experiments"
                    ),
                    source="improvement_ranking",
                ))

    except Exception as exc:
        logger.debug("[WER] _build_key_alerts failed: %s", exc)
        alerts.append(Alert(
            severity=ALERT_WARNING,
            message=f"Alert builder error: {exc}",
            source="alert_builder",
        ))

    # Sort: CRITICAL first, then WARNING, then INFO
    _order = {ALERT_CRITICAL: 0, ALERT_WARNING: 1, ALERT_INFO: 2}
    alerts.sort(key=lambda a: _order.get(a.severity, 3))

    return KeyAlertsSection(
        alerts=alerts,
        critical_count=sum(1 for a in alerts if a.severity == ALERT_CRITICAL),
        warning_count=sum(1 for a in alerts if a.severity == ALERT_WARNING),
        info_count=sum(1 for a in alerts if a.severity == ALERT_INFO),
    )


# ── Serialization ─────────────────────────────────────────────────────────────

def _review_to_dict(review: WeeklyExecutiveReview) -> dict:
    """Convert WeeklyExecutiveReview to a plain dict for JSON output."""
    h = review.system_health
    e = review.experiment_summary
    r = review.improvement_ranking
    c = review.capital_deployment
    rb = review.rollback_activity
    p = review.performance_summary
    k = review.key_alerts

    return {
        "review_date":    review.review_date,
        "lookback_days":  review.lookback_days,
        "schema_version": review.schema_version,

        "system_health": {
            "overall_status":       h.overall_status,
            "check_count":          h.check_count,
            "warning_count":        h.warning_count,
            "critical_count":       h.critical_count,
            "stale_telemetry":      h.stale_telemetry,
            "missing_telemetry":    h.missing_telemetry,
            "orphaned_experiments": h.orphaned_experiments,
        },

        "experiment_summary": {
            "total_active":          e.total_active,
            "active_experiments": [
                {
                    "experiment_id":    x.experiment_id,
                    "status":           x.status,
                    "allocation_level": x.allocation_level,
                    "allocation_pct":   x.allocation_pct,
                    "start_date":       x.start_date,
                }
                for x in e.active_experiments
            ],
            "escalated_this_week":   e.escalated_this_week,
            "rollback_candidates":   e.rollback_candidates,
            "rolled_back_this_week": e.rolled_back_this_week,
        },

        "improvement_ranking": {
            "total_ranked": r.total_ranked,
            "top_10": [
                {
                    "rank": x.rank, "experiment_id": x.experiment_id,
                    "ranking_score": x.ranking_score,
                    "total_pnl": x.total_pnl, "win_rate": x.win_rate,
                    "max_drawdown": x.max_drawdown,
                }
                for x in r.top_10
            ],
            "bottom_10": [
                {
                    "rank": x.rank, "experiment_id": x.experiment_id,
                    "ranking_score": x.ranking_score,
                    "total_pnl": x.total_pnl, "win_rate": x.win_rate,
                    "max_drawdown": x.max_drawdown,
                }
                for x in r.bottom_10
            ],
        },

        "capital_deployment": {
            "total_allocated_pct":   c.total_allocated_pct,
            "allocation_by_experiment": c.allocation_by_experiment,
            "changes_this_week": [
                {
                    "experiment_id":  ch.experiment_id,
                    "date":           ch.date,
                    "decision":       ch.decision,
                    "previous_level": ch.previous_level,
                    "new_level":      ch.new_level,
                    "allocation_pct": ch.allocation_pct,
                }
                for ch in c.changes_this_week
            ],
        },

        "rollback_activity": {
            "experiments_stopped":    rb.experiments_stopped,
            "rollback_candidate_ids": rb.rollback_candidate_ids,
            "events_this_week": [
                {
                    "experiment_id":   ev.experiment_id,
                    "date":            ev.date,
                    "decision":        ev.decision,
                    "trigger_reasons": ev.trigger_reasons,
                }
                for ev in rb.events_this_week
            ],
        },

        "performance_summary": {
            "aggregate_pnl":   p.aggregate_pnl,
            "avg_expectancy":  p.avg_expectancy,
            "avg_drawdown":    p.avg_drawdown,
            "avg_win_rate":    p.avg_win_rate,
            "total_trades":    p.total_trades,
            "experiment_count": p.experiment_count,
        },

        "key_alerts": {
            "critical_count": k.critical_count,
            "warning_count":  k.warning_count,
            "info_count":     k.info_count,
            "alerts": [
                {
                    "severity":      a.severity,
                    "message":       a.message,
                    "experiment_id": a.experiment_id,
                    "source":        a.source,
                }
                for a in k.alerts
            ],
        },
    }


def write_review(review: WeeklyExecutiveReview, path: Path) -> None:
    """Atomically overwrite weekly_executive_review.json. FAIL_OPEN."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(_review_to_dict(review), ensure_ascii=False, indent=2, default=str)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tf:
                tf.write(data)
                tf.flush()
                os.fsync(tf.fileno())
        except Exception:
            try:
                os.unlink(tmp)
            except Exception:
                pass
            raise
        Path(tmp).replace(path)
    except Exception as exc:
        logger.warning("[WER] write_review failed: %s", exc)


def load_review(path: Path) -> Optional[dict]:
    """Load the weekly_executive_review.json snapshot. FAIL_OPEN → None."""
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.debug("[WER] load_review %s: %s", path, exc)
        return None


# ── Main entry point ──────────────────────────────────────────────────────────

def run_weekly_executive_review(
    evidence_file:     Path,
    shadow_file:       Path,
    promotion_file:    Path,
    gate_file:         Path,
    registry_file:     Path,
    escalation_file:   Path,
    rollback_file:     Path,
    ranking_file:      Path,
    health_report_file: Path,
    performance_file:  Path,
    output_file:       Path,
    today:             str,
) -> WeeklyExecutiveReview:
    """
    Build the weekly executive review from all telemetry sources and write JSON.

    Never raises. FAIL_OPEN on every section.
    """
    try:
        td = _today_naive(today)

        # Load data
        health_report    = _load_json(health_report_file)
        registry         = _load_registry(registry_file)
        escalation_rows  = _load_jsonl(escalation_file)
        rollback_rows    = _load_jsonl(rollback_file)
        ranking_rows     = _load_jsonl(ranking_file)
        perf_rows        = _load_jsonl(performance_file)

        # Build sections (each independently fail-open)
        health  = _build_system_health(health_report)
        exp_sum = _build_experiment_summary(registry, escalation_rows, rollback_rows, td)
        ranking = _build_improvement_ranking(ranking_rows)
        capital = _build_capital_deployment(registry, escalation_rows, td)
        rollback = _build_rollback_activity(rollback_rows, td)
        perf    = _build_performance_summary(perf_rows, td)
        alerts  = _build_key_alerts(health, exp_sum, rollback, perf, ranking, capital)

        review = WeeklyExecutiveReview(
            review_date=today,
            lookback_days=LOOKBACK_DAYS,
            system_health=health,
            experiment_summary=exp_sum,
            improvement_ranking=ranking,
            capital_deployment=capital,
            rollback_activity=rollback,
            performance_summary=perf,
            key_alerts=alerts,
        )
        write_review(review, output_file)
        return review

    except Exception as exc:
        logger.warning("[WER] run_weekly_executive_review failed: %s", exc)
        empty = _empty_review(today, str(exc))
        try:
            write_review(empty, output_file)
        except Exception:
            pass
        return empty


def _empty_review(today: str, error: str = "") -> WeeklyExecutiveReview:
    return WeeklyExecutiveReview(
        review_date=today,
        lookback_days=LOOKBACK_DAYS,
        system_health=SystemHealthSection(
            overall_status="UNKNOWN", stale_telemetry=[], missing_telemetry=[],
            orphaned_experiments=[], check_count=0, warning_count=0, critical_count=0,
        ),
        experiment_summary=ExperimentSummarySection(
            active_experiments=[], escalated_this_week=[],
            rollback_candidates=[], rolled_back_this_week=[], total_active=0,
        ),
        improvement_ranking=ImprovementRankingSection(top_10=[], bottom_10=[], total_ranked=0),
        capital_deployment=CapitalDeploymentSection(
            allocation_by_experiment={}, changes_this_week=[], total_allocated_pct=0.0,
        ),
        rollback_activity=RollbackActivitySection(
            events_this_week=[], experiments_stopped=[], rollback_candidate_ids=[],
        ),
        performance_summary=PerformanceSummarySection(
            aggregate_pnl=0.0, avg_expectancy=0.0, avg_drawdown=0.0,
            avg_win_rate=0.0, total_trades=0, experiment_count=0,
        ),
        key_alerts=KeyAlertsSection(
            alerts=[Alert(ALERT_WARNING, f"Review runner error: {error}", source="runner")]
            if error else [],
            critical_count=0, warning_count=1 if error else 0, info_count=0,
        ),
    )


# ── Report formatter ──────────────────────────────────────────────────────────

def format_executive_review(review: WeeklyExecutiveReview) -> str:
    """Return a compact human-readable weekly executive review string."""
    lines: List[str] = []
    sep  = "═" * 72
    sep2 = "─" * 72

    _icon = {"HEALTHY": "✓", "WARNING": "△", "CRITICAL": "✗", "UNKNOWN": "?"}

    lines.append(sep)
    lines.append(
        f"WEEKLY EXECUTIVE REVIEW  [{review.review_date}]"
        f"  (last {review.lookback_days}d)"
    )
    lines.append(sep2)

    # 1. System Health
    h = review.system_health
    ic = _icon.get(h.overall_status, "?")
    lines.append(f"  [{ic}] SYSTEM HEALTH: {h.overall_status}"
                 f"  ({h.warning_count}W/{h.critical_count}C of {h.check_count} checks)")
    if h.stale_telemetry:
        lines.append(f"       Stale: {', '.join(h.stale_telemetry)}")
    if h.missing_telemetry:
        lines.append(f"       Missing: {', '.join(h.missing_telemetry)}")
    lines.append(sep2)

    # 2. Experiment Summary
    e = review.experiment_summary
    lines.append(f"  EXPERIMENTS  active={e.total_active}"
                 f"  escalated={len(e.escalated_this_week)}"
                 f"  rb_candidates={len(e.rollback_candidates)}"
                 f"  stopped={len(e.rolled_back_this_week)}")
    for ex in e.active_experiments:
        lines.append(
            f"    {ex.experiment_id:<22} L{ex.allocation_level}"
            f" ({ex.allocation_pct:.1%})"
        )
    lines.append(sep2)

    # 3. Improvement Ranking
    r = review.improvement_ranking
    lines.append(f"  IMPROVEMENT RANKING  (total={r.total_ranked})")
    if r.top_10:
        lines.append("    Top:")
        for x in r.top_10[:5]:
            lines.append(
                f"      #{x.rank} {x.experiment_id:<20} score={x.ranking_score:.1f}"
                f"  pnl={x.total_pnl:+.0f}"
            )
    if r.bottom_10:
        lines.append("    Bottom:")
        for x in r.bottom_10[:3]:
            lines.append(
                f"      #{x.rank} {x.experiment_id:<20} score={x.ranking_score:.1f}"
                f"  pnl={x.total_pnl:+.0f}"
            )
    lines.append(sep2)

    # 4. Capital Deployment
    c = review.capital_deployment
    lines.append(f"  CAPITAL DEPLOYMENT  total_allocated={c.total_allocated_pct:.1%}"
                 f"  changes={len(c.changes_this_week)}")
    for ch in c.changes_this_week[:5]:
        lines.append(
            f"    {ch.experiment_id:<22} {ch.decision:<20}"
            f" L{ch.previous_level}→L{ch.new_level} ({ch.allocation_pct:.1%})"
        )
    lines.append(sep2)

    # 5. Rollback Activity
    rb = review.rollback_activity
    lines.append(f"  ROLLBACK ACTIVITY  stopped={len(rb.experiments_stopped)}"
                 f"  candidates={len(rb.rollback_candidate_ids)}")
    for ev in rb.events_this_week[:5]:
        rsn = ", ".join(ev.trigger_reasons) or "—"
        lines.append(f"    {ev.experiment_id:<22} {ev.decision:<22} [{rsn}]")
    lines.append(sep2)

    # 6. Performance
    p = review.performance_summary
    lines.append(
        f"  PERFORMANCE (7d)  pnl={p.aggregate_pnl:+.0f}"
        f"  expectancy={p.avg_expectancy:.3f}"
        f"  drawdown={p.avg_drawdown:.1%}"
        f"  win_rate={p.avg_win_rate:.1%}"
        f"  trades={p.total_trades}"
    )
    lines.append(sep2)

    # 7. Key Alerts
    k = review.key_alerts
    lines.append(f"  KEY ALERTS  {k.critical_count}✗ {k.warning_count}△ {k.info_count}ℹ")
    _sev_ic = {ALERT_CRITICAL: "✗", ALERT_WARNING: "△", ALERT_INFO: "ℹ"}
    for a in k.alerts:
        ic2 = _sev_ic.get(a.severity, "?")
        eid_str = f" [{a.experiment_id}]" if a.experiment_id else ""
        lines.append(f"    {ic2} {a.message}{eid_str}")

    lines.append(sep)
    return "\n".join(lines)
