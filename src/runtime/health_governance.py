"""
src/runtime/health_governance.py
Runtime governance: execution lock health + deployability efficiency.

Integrates:
  - execution lock health (stale recovery events, heartbeat status)
  - deployability efficiency (alloc_survival_rate, idle_cash_ratio)
  - allocator realism metrics
  - pipeline survivorship

Writes deterministic structured JSONL governance summaries.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


class RuntimeHealthGovernance:
    def __init__(
        self,
        report_path: str = "artifacts/runtime/health_reports.jsonl",
    ) -> None:
        self._path = Path(report_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ── core assessment ───────────────────────────────────────────────────────

    def assess(
        self,
        recovery_log_path:     str,
        decision_ledger_path:  str,
        lock_recovery_log:     str = "",
        deployability_log:     str = "",
        lock_path:             str = "",
    ) -> Dict[str, Any]:
        """
        Assess runtime health across:
          - pipeline survivorship (decision ledger)
          - execution lock health (stale recovery events)
          - deployability efficiency (capital utilization, alloc survival)

        Returns health report dict and appends to JSONL report log.
        """
        decisions  = _load_jsonl(decision_ledger_path)
        recoveries = _load_jsonl(recovery_log_path)
        lock_events = _load_jsonl(lock_recovery_log) if lock_recovery_log else []
        deploy_recs = _load_jsonl(deployability_log)  if deployability_log  else []

        # ── pipeline survivorship ─────────────────────────────────────────────
        total  = sum(1 for d in decisions if d.get("decision_type") == "run_started")
        failed = sum(1 for d in decisions if d.get("decision_type") == "run_failed")
        retries = sum(1 for r in recoveries if r.get("event") != "recovered")
        survivorship = (total - failed) / total if total else 1.0

        # ── lock health ───────────────────────────────────────────────────────
        lock_health = self._assess_lock_health(lock_events, lock_path)

        # ── deployability efficiency ──────────────────────────────────────────
        deploy_health = self._assess_deployability(deploy_recs)

        # ── composite health flag ─────────────────────────────────────────────
        pipeline_ok    = (failed / total < 0.2) if total else True
        lock_ok        = lock_health["lock_status"] in ("clean", "recovered")
        deployability_ok = deploy_health["alloc_survival_rate"] >= 0.5

        report: Dict[str, Any] = {
            "ts":                     time.time(),
            "total_runs":             total,
            "failed_runs":            failed,
            "pipeline_survivorship":  round(survivorship, 4),
            "retry_concentration":    round(retries / total, 4) if total else 0.0,
            "health_ok":              pipeline_ok and lock_ok,
            # lock section
            "lock_health":            lock_health,
            # deployability section
            "deployability_health":   deploy_health,
            "deployability_ok":       deployability_ok,
        }

        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(report) + "\n")

        return report

    # ── lock health ───────────────────────────────────────────────────────────

    def _assess_lock_health(
        self,
        lock_events: list[dict],
        lock_path:   str,
    ) -> Dict[str, Any]:
        stale_recoveries = [
            e for e in lock_events if e.get("event") == "stale_lock_recovered"
        ]
        recent_recoveries = [
            e for e in stale_recoveries
            if time.time() - e.get("ts", 0) < 86400  # last 24h
        ]
        # Check current lock state
        lock_alive    = False
        lock_pid:     Optional[int] = None
        lock_hb_age:  Optional[float] = None
        lock_mode:    Optional[str] = None
        if lock_path:
            lp = Path(lock_path)
            if lp.exists():
                try:
                    data = json.loads(lp.read_text(encoding="utf-8"))
                    lock_pid  = data.get("pid")
                    lock_mode = data.get("mode")
                    hb        = data.get("heartbeat_ts")
                    lock_hb_age = round(time.time() - hb, 1) if hb else None
                    lock_alive = True
                except Exception:
                    pass

        status = "clean"
        if lock_alive:
            status = "active"
        elif recent_recoveries:
            status = "recovered"

        return {
            "lock_status":              status,
            "lock_held":                lock_alive,
            "lock_pid":                 lock_pid,
            "lock_mode":                lock_mode,
            "lock_heartbeat_age_sec":   lock_hb_age,
            "stale_recoveries_total":   len(stale_recoveries),
            "stale_recoveries_24h":     len(recent_recoveries),
        }

    # ── deployability efficiency ──────────────────────────────────────────────

    def _assess_deployability(self, deploy_recs: list[dict]) -> Dict[str, Any]:
        if not deploy_recs:
            return {
                "alloc_survival_rate":        None,
                "capital_efficiency":         None,
                "idle_cash_ratio":            None,
                "undeployable_alpha_count":   None,
                "rejection_reason_breakdown": {},
                "sample_count":               0,
            }
        n = len(deploy_recs)
        # Averages over recorded sessions
        avg_survival   = sum(r.get("alloc_survival_rate",     0.0) for r in deploy_recs) / n
        avg_cap_eff    = sum(r.get("capital_efficiency",      0.0) for r in deploy_recs) / n
        avg_idle       = sum(r.get("idle_cash_ratio",         0.0) for r in deploy_recs) / n
        total_undeployable_alpha = sum(r.get("undeployable_alpha_count", 0) for r in deploy_recs)

        # Merge rejection breakdowns
        combined_breakdown: Dict[str, int] = {}
        for r in deploy_recs:
            for reason, cnt in r.get("rejection_reason_breakdown", {}).items():
                combined_breakdown[reason] = combined_breakdown.get(reason, 0) + cnt

        return {
            "alloc_survival_rate":        round(avg_survival, 4),
            "capital_efficiency":         round(avg_cap_eff,  4),
            "idle_cash_ratio":            round(avg_idle,     4),
            "undeployable_alpha_count":   total_undeployable_alpha,
            "rejection_reason_breakdown": combined_breakdown,
            "sample_count":               n,
        }


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_jsonl(path: str) -> list[dict]:
    fp = Path(path)
    if not fp.exists():
        return []
    records = []
    try:
        with fp.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except Exception:
        pass
    return records
