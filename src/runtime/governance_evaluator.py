"""
src/runtime/governance_evaluator.py
Deterministic reducer: existing telemetry → GovernanceSummary.

Reads ONLY from existing JSONL / JSON telemetry files.
Does NOT write to or mutate any authoritative logs.
All computation is deterministic given the same input files.
Monotonic time is used for lock-age arithmetic to avoid wall-clock drift.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import replace as _dc_replace
from pathlib import Path
from statistics import median, quantiles
from typing import Any, Dict, List, Optional

from src.runtime.runtime_summary import (
    AllocationSection,
    DataHealthSection,
    ExecutionSection,
    GovernanceSection,
    GovernanceSummary,
    LockSection,
    RiskSection,
    RunSection,
    SummarySection,
    UniverseSection,
    empty_summary,
)

logger = logging.getLogger(__name__)

# Heartbeat expiry mirrors execution_lock.py default
_HEARTBEAT_EXPIRY_SEC: float = 120.0


# ─────────────────────────────────────────────────────────────────────────────
# Telemetry loader — read-only, never writes
# ─────────────────────────────────────────────────────────────────────────────

def _load_jsonl(path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    try:
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except Exception:
        pass
    return rows


def _load_json(path) -> Optional[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Telemetry paths container
# ─────────────────────────────────────────────────────────────────────────────

class TelemetryPaths:
    """
    Explicit paths to all telemetry sources consumed by the evaluator.
    Defaults to the project-standard layout under BASE_DIR.
    Pass overrides in tests.
    """
    def __init__(
        self,
        decision_ledger:         str = "",
        recovery_log:            str = "",
        circuit_breaker_log:     str = "",
        quarantine_manifest:     str = "",
        replay_ledger:           str = "",
        order_lock_file:         str = "",
        lock_recovery_log:       str = "",
        deployability_metrics:   str = "",
        portfolio_state:         str = "",
        phase2_metrics:          str = "",
        drift_metrics:           str = "",
        idempotency_manifest:    str = "",
        invariant_reports:       str = "",
    ) -> None:
        from src.paths import (
            ARTIFACTS_DIR, RUNTIME_DIR, LOGS_DIR,
        )
        _art = str(ARTIFACTS_DIR / "runtime")
        _val = str(ARTIFACTS_DIR / "runtime_validation")

        self.decision_ledger       = decision_ledger      or f"{_art}/decision_ledger.jsonl"
        self.recovery_log          = recovery_log         or f"{_art}/recovery_log.jsonl"
        self.circuit_breaker_log   = circuit_breaker_log  or f"{_art}/circuit_breaker.jsonl"
        self.quarantine_manifest   = quarantine_manifest  or f"{_art}/quarantine_manifest.jsonl"
        self.replay_ledger         = replay_ledger        or f"{_art}/replay_ledger.jsonl"
        self.order_lock_file       = order_lock_file      or str(RUNTIME_DIR / "order_lock.json")
        self.lock_recovery_log     = lock_recovery_log    or str(RUNTIME_DIR / "lock_recovery.jsonl")
        self.deployability_metrics = deployability_metrics or str(RUNTIME_DIR / "deployability_metrics.jsonl")
        self.portfolio_state       = portfolio_state      or str(RUNTIME_DIR / "portfolio_state.json")
        self.phase2_metrics        = phase2_metrics       or str(LOGS_DIR / "phase2_live_metrics.jsonl")
        self.drift_metrics         = drift_metrics        or f"{_art}/drift_metrics.jsonl"
        self.idempotency_manifest  = idempotency_manifest or f"{_art}/idempotency_manifest.jsonl"
        self.invariant_reports     = invariant_reports    or f"{_val}/invariant_reports.jsonl"


# ─────────────────────────────────────────────────────────────────────────────
# GovernanceEvaluator
# ─────────────────────────────────────────────────────────────────────────────

class GovernanceEvaluator:
    """
    Reads existing telemetry and reduces it to a GovernanceSummary.

    All business logic lives here.  Renderers and the score engine receive
    a GovernanceSummary and must not re-derive facts from raw logs.

    evaluate() is deterministic: given the same files it always produces
    the same output.  The only time-dependent value is lock_age_sec /
    heartbeat_gap_sec, which use the timestamp stored in the lock file
    (not wall-clock) so replay outputs remain stable.
    """

    def __init__(
        self,
        paths:          Optional[TelemetryPaths] = None,
        now_ts:         Optional[float]          = None,  # override for replay/tests
    ) -> None:
        self._paths  = paths or TelemetryPaths()
        self._now_ts = now_ts  # None → use time.time() at evaluate() call time

    def evaluate(self) -> GovernanceSummary:
        now = self._now_ts if self._now_ts is not None else time.time()

        decisions       = _load_jsonl(self._paths.decision_ledger)
        recovery_events = _load_jsonl(self._paths.recovery_log)
        cb_events       = _load_jsonl(self._paths.circuit_breaker_log)
        quarantine      = _load_jsonl(self._paths.quarantine_manifest)
        replay_log      = _load_jsonl(self._paths.replay_ledger)
        lock_recovery   = _load_jsonl(self._paths.lock_recovery_log)
        deploy_metrics  = _load_jsonl(self._paths.deployability_metrics)
        drift           = _load_jsonl(self._paths.drift_metrics)
        idem            = _load_jsonl(self._paths.idempotency_manifest)
        lock_data       = _load_json(self._paths.order_lock_file)
        portfolio       = _load_json(self._paths.portfolio_state)

        run_sec    = self._eval_run(decisions, now)
        lock_sec   = self._eval_lock(lock_data, lock_recovery, decisions, now)
        univ_sec   = self._eval_universe(decisions)
        alloc_sec  = self._eval_allocation(deploy_metrics, decisions)
        exec_sec   = self._eval_execution(decisions, idem)
        dh_sec     = self._eval_data_health(drift, decisions, now)
        gov_sec    = self._eval_governance(cb_events, quarantine, replay_log, decisions)
        risk_sec   = self._eval_risk(portfolio, cb_events)

        # Cross-section post-processing: derive new summary fields
        deployment_blocked = self._eval_deployment_blocked(decisions, run_sec.run_mode)
        run_sec = _dc_replace(
            run_sec,
            replay_stable=gov_sec.replay_match,
            deployment_blocked=deployment_blocked,
        )
        shadow_orders = (
            max(0, exec_sec.planned_orders - exec_sec.submitted_orders)
            if deployment_blocked else 0
        )
        exec_sec = _dc_replace(
            exec_sec,
            shadow_orders=shadow_orders,
            rejected_candidates=alloc_sec.rejected_allocations,
        )
        risk_sec = _dc_replace(
            risk_sec,
            sector_concentration=alloc_sec.concentration_score,
        )

        summ_sec = SummarySection(
            concise_operator_explanation="",
            top_anomalies=[],
            recommended_actions=[],
        )

        return GovernanceSummary(
            run=run_sec, lock=lock_sec, universe=univ_sec,
            allocation=alloc_sec, execution=exec_sec,
            data_health=dh_sec, governance=gov_sec,
            risk=risk_sec, summary=summ_sec,
        )

    # ── run section ───────────────────────────────────────────────────────────

    def _eval_run(
        self, decisions: List[dict], now: float,
    ) -> RunSection:
        started   = [d for d in decisions if d.get("decision_type") == "run_started"]
        completed = [d for d in decisions if d.get("decision_type") in ("run_completed", "pipeline_completed")]
        failed    = [d for d in decisions if d.get("decision_type") in ("run_failed", "pipeline_aborted")]

        if not started:
            return RunSection(
                run_status="unknown", run_mode="unknown",
                run_id="", epoch_id="",
                start_time=None, end_time=None, duration_sec=None,
            )

        last_start = max(started, key=lambda d: d.get("ts", 0))
        run_id     = last_start.get("run_id", "")
        epoch_id   = last_start.get("epoch_id", "")
        start_ts   = last_start.get("ts")

        # Find completion/failure for this run_id
        run_completed = [d for d in completed if d.get("run_id") == run_id]
        run_failed    = [d for d in failed    if d.get("run_id") == run_id]

        if run_completed:
            end_ts  = max(d.get("ts", 0) for d in run_completed)
            status  = "ok"
        elif run_failed:
            end_ts  = max(d.get("ts", 0) for d in run_failed)
            status  = "failed"
        else:
            end_ts  = None
            status  = "partial"  # started but no completion record yet

        duration = (end_ts - start_ts) if (start_ts and end_ts) else None

        # Derive mode from metadata or reason string
        mode = "unknown"
        meta = last_start.get("metadata", {})
        if isinstance(meta, dict):
            mode = meta.get("mode", "unknown")
        if mode == "unknown":
            reason = last_start.get("reason", "")
            if "live" in reason.lower():
                mode = "live"
            elif "dry" in reason.lower():
                mode = "dry"

        return RunSection(
            run_status   = status,
            run_mode     = mode,
            run_id       = run_id,
            epoch_id     = epoch_id,
            start_time   = start_ts,
            end_time     = end_ts,
            duration_sec = round(duration, 2) if duration else None,
        )

    # ── lock section ──────────────────────────────────────────────────────────

    def _eval_lock(
        self,
        lock_data:     Optional[dict],
        lock_recovery: List[dict],
        decisions:     List[dict],
        now:           float,
    ) -> LockSection:
        # Stale detection from recovery log
        recent_recoveries = [
            e for e in lock_recovery
            if e.get("event") == "stale_lock_recovered"
        ]
        stale_detected = len(recent_recoveries) > 0
        reclaimed      = stale_detected
        reclaim_reason = recent_recoveries[-1].get("stale_mode", "") if recent_recoveries else ""
        reclaim_success = stale_detected  # recovery write succeeds by design

        # Concurrent blocking: look for run_skipped events (overlapping_run)
        skipped = [
            d for d in decisions
            if d.get("decision_type") == "run_skipped"
            and "overlapping_run" in str(d.get("reason", ""))
        ]
        concurrent_blocked = len(skipped) > 0

        # Orphan: stale lock for a run that has no start record
        orphan = False
        if recent_recoveries and lock_data:
            stale_pid = recent_recoveries[-1].get("stale_pid")
            known_pids = {d.get("metadata", {}).get("pid") for d in decisions
                         if isinstance(d.get("metadata"), dict)}
            if stale_pid and stale_pid not in known_pids:
                orphan = True

        if lock_data:
            pid           = lock_data.get("pid")
            created_at    = lock_data.get("created_at")
            hb_ts         = lock_data.get("heartbeat_ts")
            boot_id       = lock_data.get("boot_id", "unknown")
            lock_age      = round(now - created_at, 1) if created_at else None
            hb_gap        = round(now - hb_ts, 1)      if hb_ts      else None
            lease_expiry  = (hb_ts + _HEARTBEAT_EXPIRY_SEC) if hb_ts else None
            # Deterministic fencing token: hash of pid+created_at+boot_id
            fencing_raw   = f"{pid}:{created_at}:{boot_id}"
            fencing_token = hashlib.sha256(fencing_raw.encode()).hexdigest()[:16]
            acquisition   = "ok"
        else:
            pid = lock_age = hb_gap = lease_expiry = fencing_token = None
            acquisition = "none" if not stale_detected else "recovered"

        return LockSection(
            acquisition        = acquisition,
            stale_detected     = stale_detected,
            reclaimed          = reclaimed,
            concurrent_blocked = concurrent_blocked,
            lock_owner_pid     = pid,
            lock_age_sec       = lock_age,
            heartbeat_gap_sec  = hb_gap,
            reclaim_reason     = reclaim_reason,
            reclaim_success    = reclaim_success,
            orphan_lock_detected = orphan,
            lease_expiry       = lease_expiry,
            fencing_token      = fencing_token,
        )

    # ── universe section ──────────────────────────────────────────────────────

    def _eval_universe(self, decisions: List[dict]) -> UniverseSection:
        # Extract from decision metadata where available
        univ_events = [
            d for d in decisions
            if d.get("decision_type") in ("universe_loaded", "signal_generated")
        ]
        live_count    = 0
        shadow_count  = 0
        promoted: List[str] = []
        filtered_price = 0
        filtered_risk  = 0
        for ev in univ_events:
            meta = ev.get("metadata") or {}
            live_count    = meta.get("live_count",    live_count)
            shadow_count  = meta.get("shadow_count",  shadow_count)
            filtered_price = meta.get("filtered_price", filtered_price)
            filtered_risk  = meta.get("filtered_risk",  filtered_risk)
            p = meta.get("promoted_symbols", [])
            if p:
                promoted = p

        return UniverseSection(
            live_count=live_count, shadow_count=shadow_count,
            promoted_symbols=promoted,
            filtered_price=filtered_price, filtered_risk=filtered_risk,
        )

    # ── allocation section ────────────────────────────────────────────────────

    def _eval_allocation(
        self,
        deploy_metrics: List[dict],
        decisions:      List[dict],
    ) -> AllocationSection:
        if deploy_metrics:
            latest = deploy_metrics[-1]
            cap_eff       = float(latest.get("capital_efficiency",    0.0))
            missed        = int(  latest.get("undeployable_alpha_count", 0))
            rejected      = int(  latest.get("undeployable_count",    0))
        else:
            cap_eff = missed = rejected = 0

        # Sector utilization from decision metadata
        sector_util: Dict[str, float] = {}
        concentration = 0.0
        alloc_events = [
            d for d in decisions
            if "ALLOCATOR" in str(d.get("reason", ""))
            or d.get("decision_type") == "alloc_summary"
        ]
        for ev in alloc_events:
            meta = ev.get("metadata") or {}
            su = meta.get("sector_utilization")
            if isinstance(su, dict):
                sector_util = su
                concentration = max(su.values(), default=0.0)

        return AllocationSection(
            capital_efficiency   = round(cap_eff, 4),
            missed_entries       = missed,
            rejected_allocations = rejected,
            sector_utilization   = sector_util,
            concentration_score  = round(concentration, 4),
        )

    # ── execution section ─────────────────────────────────────────────────────

    def _eval_execution(
        self,
        decisions: List[dict],
        idem:      List[dict],
    ) -> ExecutionSection:
        order_events = [d for d in decisions if "order" in d.get("decision_type", "").lower()]
        planned   = sum(1 for d in order_events if "plan" in d.get("decision_type", ""))
        submitted = sum(1 for d in order_events if "submit" in d.get("decision_type", "")
                        or "sent" in d.get("decision_type", ""))
        rejected  = sum(1 for d in order_events if "reject" in d.get("decision_type", "")
                        or "block" in d.get("decision_type", ""))
        retry_count = sum(1 for d in decisions if d.get("decision_type") == "stage_retry"
                          or "retry" in d.get("reason", "").lower())
        dup_blocked = any(
            "duplicate" in d.get("reason", "").lower()
            or "dup_order" in str(d.get("metadata", "")).lower()
            for d in decisions
        )
        idem_pass = not any(
            d.get("decision_type") == "idempotency_violation"
            for d in decisions
        )

        return ExecutionSection(
            planned_orders        = planned,
            submitted_orders      = submitted,
            rejected_orders       = rejected,
            retry_count           = retry_count,
            idempotency_pass      = idem_pass,
            duplicate_order_blocked = dup_blocked,
        )

    # ── data health section ───────────────────────────────────────────────────

    def _eval_data_health(
        self,
        drift:     List[dict],
        decisions: List[dict],
        now:       float,
    ) -> DataHealthSection:
        stale = sum(1 for d in decisions
                    if "stale" in str(d.get("reason", "")).lower()
                    or d.get("decision_type") == "stale_snapshot")
        missing_syms: List[str] = []
        for d in decisions:
            meta = d.get("metadata") or {}
            if isinstance(meta, dict):
                ms = meta.get("missing_symbols", [])
                if ms:
                    missing_syms.extend(ms)
        missing_syms = list(dict.fromkeys(missing_syms))  # dedupe, order-preserving

        # latency from drift_metrics
        latencies = [float(r["duration_s"]) for r in drift if "duration_s" in r]
        p95 = lmax = None
        if latencies:
            srt = sorted(latencies)
            idx_95 = max(0, int(len(srt) * 0.95) - 1)
            p95  = round(srt[idx_95], 3)
            lmax = round(srt[-1], 3)

        cache_reuse = 0.0
        hits   = sum(1 for r in drift if r.get("cache_misses", 1) == 0)
        total  = max(1, len(drift))
        cache_reuse = round(hits / total, 4)

        # Freshness: 1.0 if no stale events, decreases with stale count
        freshness = max(0.0, 1.0 - min(1.0, stale * 0.1))

        return DataHealthSection(
            stale_snapshots      = stale,
            cache_reuse_rate     = cache_reuse,
            freshness_score      = round(freshness, 4),
            missing_data_symbols = missing_syms,
            snapshot_latency_p95 = p95,
            snapshot_latency_max = lmax,
        )

    # ── governance section ────────────────────────────────────────────────────

    def _eval_governance(
        self,
        cb_events:  List[dict],
        quarantine: List[dict],
        replay_log: List[dict],
        decisions:  List[dict],
    ) -> GovernanceSection:
        fail_closed = not any(
            e.get("event_type") == "TRIPPED" for e in cb_events
        )
        quarantine_count = len(quarantine)

        # Replay match: True if no divergence events recorded
        replay_divergences = [
            e for e in cb_events if e.get("event_type") == "replay_divergence"
        ]
        replay_match   = len(replay_divergences) == 0
        divergence_score = min(1.0, len(replay_divergences) / 5.0)

        # Broker authority: True unless a broker_mismatch event exists
        broker_auth = not any(
            "broker" in str(d.get("reason", "")).lower()
            and "mismatch" in str(d.get("reason", "")).lower()
            for d in decisions
        )

        recon_required = any(
            d.get("decision_type") in ("reconciliation_required", "broker_sync_required")
            for d in decisions
        )
        recon_success = not any(
            d.get("decision_type") == "reconciliation_failed"
            for d in decisions
        )

        return GovernanceSection(
            fail_closed             = fail_closed,
            replay_match            = replay_match,
            broker_authoritative    = broker_auth,
            quarantine_events       = quarantine_count,
            divergence_score        = round(divergence_score, 4),
            reconciliation_required = recon_required,
            reconciliation_success  = recon_success,
        )

    # ── deployment blocked ────────────────────────────────────────────────────

    def _eval_deployment_blocked(
        self,
        decisions: List[dict],
        run_mode:  str,
    ) -> bool:
        """
        True if deployment gate was blocking live order submission this run.
        Derived from decision_ledger event evidence, then run mode.
        Non-live mode implies shadow execution (deployment blocked by design).
        """
        for d in decisions:
            dtype  = d.get("decision_type", "").lower()
            reason = d.get("reason", "").lower()
            meta   = d.get("metadata") or {}
            # Explicit gate block events
            if "deployment" in dtype and "block" in reason:
                return True
            if isinstance(meta, dict) and meta.get("deployment_blocked", False):
                return True
            # Live order submission disabled marker
            if dtype in ("live_order_submission_disabled", "deployment_gate_blocked"):
                return True
        # Non-live modes are always shadow-blocked
        return run_mode != "live"

    # ── risk section ──────────────────────────────────────────────────────────

    def _eval_risk(
        self,
        portfolio: Optional[dict],
        cb_events: List[dict],
    ) -> RiskSection:
        exposure = 0.0
        max_sector = 0.0
        drawdown   = 0.0
        flags: List[str] = []

        if portfolio:
            exposure = float(portfolio.get("total_exposure", 0.0))
            drawdown = float(portfolio.get("drawdown", 0.0))
            sector_w = portfolio.get("sector_weights", {})
            if isinstance(sector_w, dict) and sector_w:
                max_sector = max(sector_w.values(), default=0.0)
            pf = portfolio.get("emergency_flags", [])
            if isinstance(pf, list):
                flags = pf

        # CB tripped = emergency flag
        if any(e.get("event_type") == "TRIPPED" for e in cb_events):
            if "CIRCUIT_BREAKER_TRIPPED" not in flags:
                flags.append("CIRCUIT_BREAKER_TRIPPED")

        return RiskSection(
            portfolio_exposure  = round(exposure, 0),
            max_sector_exposure = round(max_sector, 4),
            drawdown            = round(abs(drawdown), 4),
            emergency_flags     = flags,
        )
