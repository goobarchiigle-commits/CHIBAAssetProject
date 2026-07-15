"""
src/runtime/summary_pipeline.py
End-of-run governance compression pipeline.

Pipeline:
  raw telemetry
  -> GovernanceEvaluator  (deterministic reducer)
  -> SeverityClassifier   (invariant annotation)
  -> RuntimeScoreEngine   (deterministic 0-100 score)
  -> SummaryComposer      (fills summary section)
  -> JSONRenderer         (zero-logic JSON output)
  -> MarkdownRenderer     (zero-logic Markdown output)
  -> atomic write to reports/runtime_summary/latest.*

Writes:
  reports/runtime_summary/latest.json
  reports/runtime_summary/latest.md
  reports/execution_health/latest.json
  reports/governance_summary/latest.md

Does NOT mutate any existing telemetry.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import replace as _dc_replace
from pathlib import Path
from typing import Optional

from src.runtime.governance_evaluator import GovernanceEvaluator, TelemetryPaths
from src.runtime.json_renderer import JSONRenderer
from src.runtime.markdown_renderer import MarkdownRenderer
from src.runtime.runtime_score import RuntimeScoreEngine, score_band
from src.runtime.runtime_summary import (
    CRITICAL, WARN,
    AnomalyEvent,
    GovernanceSummary,
    SummarySection,
)
from src.runtime.severity_classifier import SeverityClassifier

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SummaryComposer — fills SummarySection from anomalies + score, zero logic
# ─────────────────────────────────────────────────────────────────────────────

class SummaryComposer:
    """Assembles the human-readable SummarySection from classified anomalies."""

    def compose(
        self,
        summary:   GovernanceSummary,
        anomalies: list[AnomalyEvent],
        score:     int,
    ) -> SummarySection:
        criticals = [a for a in anomalies if a.severity == CRITICAL]
        warns     = [a for a in anomalies if a.severity == WARN]

        # Concise explanation
        if not anomalies:
            explanation = (
                f"Run completed cleanly. Score {score}/100 ({score_band(score)}). "
                "No anomalies detected."
            )
        elif criticals:
            codes = ", ".join(a.code for a in criticals[:3])
            explanation = (
                f"Run {summary.run.run_status}. Score {score}/100 ({score_band(score)}). "
                f"{len(criticals)} CRITICAL anomaly(s): {codes}. "
                "Operator intervention may be required."
            )
        else:
            codes = ", ".join(a.code for a in warns[:3])
            explanation = (
                f"Run {summary.run.run_status}. Score {score}/100 ({score_band(score)}). "
                f"{len(warns)} warning(s): {codes}."
            )

        # Top anomalies list (CRITICAL first, then WARN)
        top = [f"[{a.severity}] {a.code}: {a.description}" for a in (criticals + warns)[:5]]

        # Recommended actions
        actions = _derive_actions(criticals, warns, summary)

        return SummarySection(
            concise_operator_explanation = explanation,
            top_anomalies                = top,
            recommended_actions          = actions,
        )


def _derive_actions(
    criticals: list[AnomalyEvent],
    warns:     list[AnomalyEvent],
    summary:   GovernanceSummary,
) -> list[str]:
    actions = []
    codes = {a.code for a in criticals + warns}

    if "REPLAY_DIVERGENCE" in codes:
        actions.append("Investigate replay divergence — compare forensic logs with replay ledger.")
    if "BROKER_AUTHORITY_MISMATCH" in codes:
        actions.append("Run position_reconcile before next live execution.")
    if "STALE_LOCK_RECLAIMED" in codes or "STALE_LOCK_NOT_RECLAIMED" in codes:
        actions.append("Review lock_recovery.jsonl for stale lock root cause.")
    if "IDEMPOTENCY_VIOLATION" in codes:
        actions.append("Halt execution and audit idempotency_manifest for duplicate runs.")
    if "FAIL_OPEN_DETECTED" in codes:
        actions.append("Circuit breaker tripped — review circuit_breaker.jsonl and reset manually.")
    if "DRAWDOWN_LIMIT_BREACH" in codes:
        actions.append("Portfolio drawdown at -15% threshold. No new BUY orders until recovery.")
    if "RUN_FAILED" in codes:
        actions.append(f"Run failed. Review decision_ledger.jsonl for stage failure details.")
    if "RECONCILIATION_FAILED" in codes:
        actions.append("Manual broker reconciliation required.")
    if "LOW_CAPITAL_EFFICIENCY" in codes or "EXCESSIVE_MISSED_ENTRIES" in codes:
        actions.append("Review deployability_metrics.jsonl for idle capital root cause.")

    return actions


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class RuntimeSummaryPipeline:
    """
    Orchestrates the full governance compression pipeline.

    run() reads telemetry, produces summaries, writes reports.
    All outputs are deterministic given the same input telemetry.
    """

    def __init__(
        self,
        telemetry_paths: Optional[TelemetryPaths] = None,
        reports_dir:     Optional[Path]           = None,
        now_ts:          Optional[float]          = None,
    ) -> None:
        self._paths      = telemetry_paths
        self._reports    = reports_dir or _default_reports_dir()
        self._now_ts     = now_ts
        self._evaluator  = GovernanceEvaluator(paths=telemetry_paths, now_ts=now_ts)
        self._classifier = SeverityClassifier()
        self._scorer     = RuntimeScoreEngine()
        self._composer   = SummaryComposer()
        self._json       = JSONRenderer()
        self._md         = MarkdownRenderer()

    def run(self) -> GovernanceSummary:
        """
        Execute the full pipeline.  Returns the completed GovernanceSummary.
        Side effect: writes report files under reports_dir.
        """
        summary   = self._evaluator.evaluate()
        anomalies = self._classifier.classify(summary)
        score     = self._scorer.compute(summary, anomalies)
        summ_sec  = self._composer.compose(summary, anomalies, score)

        # Attach computed fields
        summary.anomalies     = anomalies
        summary.runtime_score = score
        summary.summary       = summ_sec

        self._write_reports(summary)
        logger.info(
            "[GOVERNANCE_PIPELINE] run=%s score=%d band=%s "
            "criticals=%d warns=%d",
            summary.run.run_id, score, score_band(score),
            sum(1 for a in anomalies if a.severity == CRITICAL),
            sum(1 for a in anomalies if a.severity == WARN),
        )
        return summary

    def _write_reports(self, summary: GovernanceSummary) -> None:
        json_str = self._json.render(summary)
        md_str   = self._md.render(summary)

        self._atomic_write(
            self._reports / "runtime_summary" / "latest.json",
            json_str,
        )
        self._atomic_write(
            self._reports / "runtime_summary" / "latest.md",
            md_str,
        )
        # execution_health: allocation + execution + risk sections only
        exec_health = {
            "ts":         time.time(),
            "run_id":     summary.run.run_id,
            "score":      summary.runtime_score,
            "band":       score_band(summary.runtime_score),
            "allocation": self._json.render_dict(summary)["allocation"],
            "execution":  self._json.render_dict(summary)["execution"],
            "risk":       self._json.render_dict(summary)["risk"],
        }
        self._atomic_write(
            self._reports / "execution_health" / "latest.json",
            json.dumps(exec_health, indent=2, sort_keys=True),
        )
        # governance_summary: governance-focused markdown
        self._atomic_write(
            self._reports / "governance_summary" / "latest.md",
            _governance_only_md(summary, self._md),
        )

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            tmp.write_text(content, encoding="utf-8")
            tmp.replace(path)
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass


def _governance_only_md(summary: GovernanceSummary, renderer: MarkdownRenderer) -> str:
    """Focused governance Markdown: header + anomalies + governance section only."""
    return "\n".join([
        renderer._header(summary),
        renderer._operator_snapshot(summary),
        renderer._anomaly_table(summary.anomalies),
        renderer._recommended_actions(summary),
        renderer._divider(),
        renderer._governance_section(summary),
        renderer._lock_section(summary),
        renderer._footer(summary),
    ])


def _default_reports_dir() -> Path:
    try:
        from src.paths import REPORTS_DIR
        return REPORTS_DIR
    except Exception:
        return Path("reports")
