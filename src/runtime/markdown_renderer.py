"""
src/runtime/markdown_renderer.py
Pure structural renderer: GovernanceSummary → Markdown string.

ZERO business logic.
ZERO arithmetic.
ZERO conditional evaluation beyond formatting guards.

Receives a fully-evaluated GovernanceSummary and renders it as Markdown.
Top section supports 5-second operational assessment.
Compatible with Discord digests and LLM ingestion.
"""
from __future__ import annotations

import datetime
from typing import List

from src.runtime.runtime_summary import (
    CRITICAL, WARN, INFO,
    AnomalyEvent,
    GovernanceSummary,
)
from src.runtime.runtime_score import score_band


class MarkdownRenderer:
    """
    Stateless Markdown renderer.

    render() is pure: same GovernanceSummary → same Markdown string.
    Section order is fixed and operator-optimised (critical info first).
    """

    def render(self, summary: GovernanceSummary) -> str:
        parts = [
            self._header(summary),
            self._operator_snapshot(summary),
            self._anomaly_table(summary.anomalies),
            self._recommended_actions(summary),
            self._divider(),
            self._run_section(summary),
            self._lock_section(summary),
            self._governance_section(summary),
            self._allocation_section(summary),
            self._execution_section(summary),
            self._risk_section(summary),
            self._data_health_section(summary),
            self._universe_section(summary),
            self._footer(summary),
        ]
        return "\n".join(p for p in parts if p)

    # ── 5-second operator snapshot ────────────────────────────────────────────

    def _header(self, s: GovernanceSummary) -> str:
        band  = score_band(s.runtime_score)
        icon  = {"HEALTHY": "✅", "DEGRADED": "⚠️", "IMPAIRED": "🔶", "CRITICAL": "🚨"}.get(band, "❓")
        title = f"# {icon} Runtime Governance Summary — {band}"
        score_line = f"**Runtime Score: {s.runtime_score}/100** ({band})"
        status_line = f"**Run Status:** `{s.run.run_status.upper()}` | **Mode:** `{s.run.run_mode.upper()}`"
        return "\n".join([title, "", score_line, status_line])

    def _operator_snapshot(self, s: GovernanceSummary) -> str:
        lines = ["## ⚡ Operator Snapshot", ""]
        # Status row
        lines.append(f"| Field | Value |")
        lines.append(f"|---|---|")
        lines.append(f"| Run ID | `{s.run.run_id or '—'}` |")
        lines.append(f"| Epoch | `{s.run.epoch_id or '—'}` |")
        lines.append(f"| Mode | `{s.run.run_mode}` |")
        lines.append(f"| Status | `{s.run.run_status}` |")
        dur = f"{s.run.duration_sec:.1f}s" if s.run.duration_sec is not None else "—"
        lines.append(f"| Duration | {dur} |")
        lines.append(f"| Score | **{s.runtime_score}/100** ({score_band(s.runtime_score)}) |")
        lines.append(f"| Lock | `{s.lock.acquisition}` |")
        lines.append(f"| Replay Stable | `{'YES' if s.run.replay_stable else 'DIVERGED'}` |")
        lines.append(f"| Deployment Blocked | `{'YES (shadow)' if s.run.deployment_blocked else 'NO (live)'}` |")
        lines.append(f"| Broker Auth | `{'YES' if s.governance.broker_authoritative else 'NO'}` |")
        lines.append(f"| Fail Closed | `{'YES' if s.governance.fail_closed else 'NO'}` |")
        lines.append(f"| Shadow Orders | {s.execution.shadow_orders} |")
        criticals = [a for a in s.anomalies if a.severity == CRITICAL]
        lines.append(f"| CRITICAL anomalies | **{len(criticals)}** |")
        return "\n".join(lines)

    def _anomaly_table(self, anomalies: List[AnomalyEvent]) -> str:
        if not anomalies:
            return "## ✅ Anomalies\n\n_No anomalies detected._"
        lines = ["## 🔍 Anomalies", ""]
        criticals = [a for a in anomalies if a.severity == CRITICAL]
        warns     = [a for a in anomalies if a.severity == WARN]
        infos     = [a for a in anomalies if a.severity == INFO]

        lines.append("| Severity | Code | Description | Section |")
        lines.append("|---|---|---|---|")
        for a in criticals + warns + infos:
            sev_icon = {"CRITICAL": "🚨", "WARN": "⚠️", "INFO": "ℹ️"}.get(a.severity, "")
            desc = a.description.replace("|", "\\|")
            lines.append(f"| {sev_icon} {a.severity} | `{a.code}` | {desc} | {a.section} |")
        return "\n".join(lines)

    def _recommended_actions(self, s: GovernanceSummary) -> str:
        if not s.summary.recommended_actions:
            return ""
        lines = ["## 🛠 Recommended Actions", ""]
        for action in s.summary.recommended_actions:
            lines.append(f"- {action}")
        return "\n".join(lines)

    def _divider(self) -> str:
        return "---"

    # ── detail sections ───────────────────────────────────────────────────────

    def _run_section(self, s: GovernanceSummary) -> str:
        r = s.run
        start = _fmt_ts(r.start_time)
        end   = _fmt_ts(r.end_time)
        dur   = f"{r.duration_sec:.1f}s" if r.duration_sec is not None else "—"
        return "\n".join([
            "## Run",
            "",
            f"| Field | Value |",
            f"|---|---|",
            f"| run_status | `{r.run_status}` |",
            f"| run_mode | `{r.run_mode}` |",
            f"| run_id | `{r.run_id or '—'}` |",
            f"| epoch_id | `{r.epoch_id or '—'}` |",
            f"| start_time | {start} |",
            f"| end_time | {end} |",
            f"| duration_sec | {dur} |",
            f"| replay_stable | `{r.replay_stable}` |",
            f"| deployment_blocked | `{r.deployment_blocked}` |",
        ])

    def _lock_section(self, s: GovernanceSummary) -> str:
        lk = s.lock
        age  = f"{lk.lock_age_sec:.0f}s"      if lk.lock_age_sec      is not None else "—"
        hbg  = f"{lk.heartbeat_gap_sec:.0f}s" if lk.heartbeat_gap_sec is not None else "—"
        exp  = _fmt_ts(lk.lease_expiry)
        pid  = str(lk.lock_owner_pid) if lk.lock_owner_pid is not None else "—"
        tok  = lk.fencing_token or "—"
        return "\n".join([
            "## Lock",
            "",
            f"| Field | Value |",
            f"|---|---|",
            f"| acquisition | `{lk.acquisition}` |",
            f"| stale_detected | `{lk.stale_detected}` |",
            f"| reclaimed | `{lk.reclaimed}` |",
            f"| reclaim_reason | `{lk.reclaim_reason or '—'}` |",
            f"| reclaim_success | `{lk.reclaim_success}` |",
            f"| orphan_lock_detected | `{lk.orphan_lock_detected}` |",
            f"| concurrent_blocked | `{lk.concurrent_blocked}` |",
            f"| lock_owner_pid | `{pid}` |",
            f"| lock_age_sec | {age} |",
            f"| heartbeat_gap_sec | {hbg} |",
            f"| lease_expiry | {exp} |",
            f"| fencing_token | `{tok}` |",
        ])

    def _governance_section(self, s: GovernanceSummary) -> str:
        gov = s.governance
        return "\n".join([
            "## Governance",
            "",
            f"| Field | Value |",
            f"|---|---|",
            f"| fail_closed | `{gov.fail_closed}` |",
            f"| replay_match | `{gov.replay_match}` |",
            f"| broker_authoritative | `{gov.broker_authoritative}` |",
            f"| quarantine_events | {gov.quarantine_events} |",
            f"| divergence_score | {gov.divergence_score:.4f} |",
            f"| reconciliation_required | `{gov.reconciliation_required}` |",
            f"| reconciliation_success | `{gov.reconciliation_success}` |",
        ])

    def _allocation_section(self, s: GovernanceSummary) -> str:
        alloc = s.allocation
        su = ", ".join(f"{k}: {v:.3f}" for k, v in sorted(alloc.sector_utilization.items())) or "—"
        return "\n".join([
            "## Allocation",
            "",
            f"| Field | Value |",
            f"|---|---|",
            f"| capital_efficiency | {alloc.capital_efficiency:.4f} |",
            f"| missed_entries | {alloc.missed_entries} |",
            f"| rejected_allocations | {alloc.rejected_allocations} |",
            f"| concentration_score | {alloc.concentration_score:.4f} |",
            f"| sector_utilization | {su} |",
        ])

    def _execution_section(self, s: GovernanceSummary) -> str:
        ex = s.execution
        return "\n".join([
            "## Execution Shadow",
            "",
            f"| Field | Value |",
            f"|---|---|",
            f"| planned_orders | {ex.planned_orders} |",
            f"| shadow_orders | {ex.shadow_orders} |",
            f"| submitted_orders | {ex.submitted_orders} |",
            f"| rejected_orders | {ex.rejected_orders} |",
            f"| rejected_candidates | {ex.rejected_candidates} |",
            f"| retry_count | {ex.retry_count} |",
            f"| idempotency_pass | `{ex.idempotency_pass}` |",
            f"| duplicate_order_blocked | `{ex.duplicate_order_blocked}` |",
        ])

    def _risk_section(self, s: GovernanceSummary) -> str:
        risk = s.risk
        flags = ", ".join(risk.emergency_flags) if risk.emergency_flags else "none"
        return "\n".join([
            "## Risk",
            "",
            f"| Field | Value |",
            f"|---|---|",
            f"| portfolio_exposure | ¥{risk.portfolio_exposure:,.0f} |",
            f"| sector_concentration | {risk.sector_concentration:.4f} |",
            f"| max_sector_exposure | {risk.max_sector_exposure:.4f} |",
            f"| drawdown | {risk.drawdown:.4f} ({risk.drawdown:.1%}) |",
            f"| emergency_flags | {flags} |",
        ])

    def _data_health_section(self, s: GovernanceSummary) -> str:
        dh  = s.data_health
        p95 = f"{dh.snapshot_latency_p95:.3f}s" if dh.snapshot_latency_p95 is not None else "—"
        lmax = f"{dh.snapshot_latency_max:.3f}s" if dh.snapshot_latency_max is not None else "—"
        miss = ", ".join(dh.missing_data_symbols[:10]) or "none"
        return "\n".join([
            "## Data Health",
            "",
            f"| Field | Value |",
            f"|---|---|",
            f"| stale_snapshots | {dh.stale_snapshots} |",
            f"| cache_reuse_rate | {dh.cache_reuse_rate:.4f} |",
            f"| freshness_score | {dh.freshness_score:.4f} |",
            f"| snapshot_latency_p95 | {p95} |",
            f"| snapshot_latency_max | {lmax} |",
            f"| missing_data_symbols | {miss} |",
        ])

    def _universe_section(self, s: GovernanceSummary) -> str:
        univ = s.universe
        promoted = ", ".join(univ.promoted_symbols[:10]) or "none"
        return "\n".join([
            "## Universe",
            "",
            f"| Field | Value |",
            f"|---|---|",
            f"| live_count | {univ.live_count} |",
            f"| shadow_count | {univ.shadow_count} |",
            f"| filtered_price | {univ.filtered_price} |",
            f"| filtered_risk | {univ.filtered_risk} |",
            f"| promoted_symbols | {promoted} |",
        ])

    def _footer(self, s: GovernanceSummary) -> str:
        start = _fmt_ts(s.run.start_time)
        return f"\n_Generated from run started {start}. Forensic telemetry remains authoritative._"


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers — structural only
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_ts(ts) -> str:
    if ts is None:
        return "—"
    try:
        dt = datetime.datetime.fromtimestamp(float(ts), tz=datetime.timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return str(ts)
