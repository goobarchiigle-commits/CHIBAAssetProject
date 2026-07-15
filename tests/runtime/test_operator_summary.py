"""
tests/runtime/test_operator_summary.py
Operator-grade runtime summary tests.

Covers all 8 required categories:
  1. deterministic summary generation
  2. replay-stable outputs (same input → same output)
  3. UTF-8 rendering consistency
  4. missing telemetry handling
  5. renderer consistency (JSON ↔ fields, MD structure)
  6. stale lock reporting
  7. divergence reporting
  8. latest summary pointer correctness
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.runtime.runtime_summary import (
    CRITICAL, INFO, WARN,
    AnomalyEvent,
    GovernanceSummary,
    RunSection, LockSection, UniverseSection, AllocationSection,
    ExecutionSection, DataHealthSection, GovernanceSection,
    RiskSection, SummarySection,
    empty_summary,
)
from src.runtime.governance_evaluator import GovernanceEvaluator, TelemetryPaths
from src.runtime.severity_classifier import SeverityClassifier
from src.runtime.runtime_score import RuntimeScoreEngine, score_band
from src.runtime.json_renderer import JSONRenderer
from src.runtime.markdown_renderer import MarkdownRenderer
from src.runtime.summary_pipeline import RuntimeSummaryPipeline, SummaryComposer


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write_jsonl(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _make_paths(tmp: Path, **overrides) -> TelemetryPaths:
    defaults = dict(
        decision_ledger       = str(tmp / "decision_ledger.jsonl"),
        recovery_log          = str(tmp / "recovery_log.jsonl"),
        circuit_breaker_log   = str(tmp / "circuit_breaker.jsonl"),
        quarantine_manifest   = str(tmp / "quarantine.jsonl"),
        replay_ledger         = str(tmp / "replay_ledger.jsonl"),
        order_lock_file       = str(tmp / "order_lock.json"),
        lock_recovery_log     = str(tmp / "lock_recovery.jsonl"),
        deployability_metrics = str(tmp / "deployability.jsonl"),
        portfolio_state       = str(tmp / "portfolio_state.json"),
        phase2_metrics        = str(tmp / "phase2.jsonl"),
        drift_metrics         = str(tmp / "drift.jsonl"),
        idempotency_manifest  = str(tmp / "idem.jsonl"),
        invariant_reports     = str(tmp / "invariant_reports.jsonl"),
    )
    defaults.update(overrides)
    return TelemetryPaths(**defaults)


def _base_decisions(run_id: str = "run-001", mode: str = "dry") -> list:
    return [
        {
            "decision_type": "run_started",
            "run_id": run_id,
            "epoch_id": "epoch-1",
            "ts": 1_700_000_000.0,
            "metadata": {"mode": mode},
        },
        {
            "decision_type": "run_completed",
            "run_id": run_id,
            "ts": 1_700_000_060.0,
        },
    ]


def _run_pipeline(tmp: Path, decisions: list | None = None, reports_dir: Path | None = None,
                  run_id: str = "run-001") -> GovernanceSummary:
    decisions = decisions or _base_decisions(run_id=run_id)
    _write_jsonl(tmp / "decision_ledger.jsonl", decisions)
    paths = _make_paths(tmp)
    rpt   = reports_dir or (tmp / "reports")
    pipeline = RuntimeSummaryPipeline(
        telemetry_paths=paths,
        reports_dir=rpt,
        now_ts=1_700_000_120.0,
    )
    return pipeline.run()


# ═════════════════════════════════════════════════════════════════════════════
# 1. DETERMINISTIC SUMMARY GENERATION
# ═════════════════════════════════════════════════════════════════════════════

class TestDeterministicSummaryGeneration:
    """Same telemetry always produces identical GovernanceSummary."""

    def test_same_input_same_output(self, tmp_path):
        decisions = _base_decisions()
        _write_jsonl(tmp_path / "decision_ledger.jsonl", decisions)
        paths = _make_paths(tmp_path)
        ev = GovernanceEvaluator(paths=paths, now_ts=1_700_000_120.0)

        s1 = ev.evaluate()
        s2 = ev.evaluate()

        assert s1.run.run_id    == s2.run.run_id
        assert s1.run.run_status == s2.run.run_status
        assert s1.governance.replay_match  == s2.governance.replay_match
        assert s1.run.replay_stable        == s2.run.replay_stable
        assert s1.run.deployment_blocked   == s2.run.deployment_blocked

    def test_new_fields_present_in_summary(self, tmp_path):
        s = _run_pipeline(tmp_path)
        # RunSection new fields
        assert hasattr(s.run, "replay_stable")
        assert hasattr(s.run, "deployment_blocked")
        # ExecutionSection new fields
        assert hasattr(s.execution, "shadow_orders")
        assert hasattr(s.execution, "rejected_candidates")
        # RiskSection new field
        assert hasattr(s.risk, "sector_concentration")

    def test_score_deterministic(self, tmp_path):
        decisions = _base_decisions()
        _write_jsonl(tmp_path / "decision_ledger.jsonl", decisions)
        paths = _make_paths(tmp_path)
        ev = GovernanceEvaluator(paths=paths, now_ts=1_700_000_120.0)
        cl = SeverityClassifier()
        sc = RuntimeScoreEngine()

        s = ev.evaluate()
        a = cl.classify(s)
        score1 = sc.compute(s, a)
        score2 = sc.compute(s, a)

        assert score1 == score2

    def test_json_output_deterministic(self, tmp_path):
        decisions = _base_decisions()
        _write_jsonl(tmp_path / "decision_ledger.jsonl", decisions)
        paths = _make_paths(tmp_path)
        ev  = GovernanceEvaluator(paths=paths, now_ts=1_700_000_120.0)
        jr  = JSONRenderer()

        s1 = ev.evaluate()
        s2 = ev.evaluate()

        assert jr.render(s1) == jr.render(s2)

    def test_markdown_output_deterministic(self, tmp_path):
        decisions = _base_decisions()
        _write_jsonl(tmp_path / "decision_ledger.jsonl", decisions)
        paths = _make_paths(tmp_path)
        ev  = GovernanceEvaluator(paths=paths, now_ts=1_700_000_120.0)
        mr  = MarkdownRenderer()

        s1 = ev.evaluate()
        s2 = ev.evaluate()

        assert mr.render(s1) == mr.render(s2)


# ═════════════════════════════════════════════════════════════════════════════
# 2. REPLAY-STABLE OUTPUTS
# ═════════════════════════════════════════════════════════════════════════════

class TestReplayStableOutputs:
    """Outputs are replay-safe: no wall-clock fields in summary JSON."""

    def test_latest_json_no_wall_clock_timestamps_in_summary(self, tmp_path):
        rpt = tmp_path / "reports"
        _run_pipeline(tmp_path, reports_dir=rpt)
        raw = (rpt / "runtime_summary" / "latest.json").read_text(encoding="utf-8")
        data = json.loads(raw)
        # Summary JSON should NOT contain a top-level 'ts' (wall-clock)
        assert "ts" not in data, "latest.json must not embed wall-clock 'ts' at root"

    def test_replay_stable_field_reflects_governance_replay_match(self, tmp_path):
        decisions = _base_decisions()
        _write_jsonl(tmp_path / "decision_ledger.jsonl", decisions)
        paths = _make_paths(tmp_path)
        ev = GovernanceEvaluator(paths=paths, now_ts=1_700_000_120.0)
        s  = ev.evaluate()
        # No replay divergence in telemetry → replay_stable must match replay_match
        assert s.run.replay_stable == s.governance.replay_match

    def test_replay_stable_false_when_divergence_exists(self, tmp_path):
        decisions = _base_decisions()
        _write_jsonl(tmp_path / "decision_ledger.jsonl", decisions)
        # Inject replay divergence circuit breaker event
        _write_jsonl(tmp_path / "circuit_breaker.jsonl", [
            {"event_type": "replay_divergence", "run_id": "run-001"},
        ])
        paths = _make_paths(tmp_path)
        ev = GovernanceEvaluator(paths=paths, now_ts=1_700_000_120.0)
        s  = ev.evaluate()
        assert s.run.replay_stable is False
        assert s.governance.replay_match is False

    def test_same_input_json_byte_identical(self, tmp_path):
        decisions = _base_decisions()
        _write_jsonl(tmp_path / "decision_ledger.jsonl", decisions)
        paths = _make_paths(tmp_path)

        ev  = GovernanceEvaluator(paths=paths, now_ts=1_700_000_120.0)
        cl  = SeverityClassifier()
        sc  = RuntimeScoreEngine()
        cmp = SummaryComposer()
        jr  = JSONRenderer()

        def _render():
            s = ev.evaluate()
            a = cl.classify(s)
            score = sc.compute(s, a)
            s.anomalies     = a
            s.runtime_score = score
            s.summary       = cmp.compose(s, a, score)
            return jr.render(s)

        assert _render() == _render()

    def test_md_footer_uses_run_start_not_wallclock(self, tmp_path):
        decisions = _base_decisions()
        _write_jsonl(tmp_path / "decision_ledger.jsonl", decisions)
        paths = _make_paths(tmp_path)
        ev = GovernanceEvaluator(paths=paths, now_ts=1_700_000_120.0)
        mr = MarkdownRenderer()
        s  = ev.evaluate()
        md = mr.render(s)
        # Footer must contain the run start timestamp string, not "Generated at"
        assert "Generated from run started" in md


# ═════════════════════════════════════════════════════════════════════════════
# 3. UTF-8 RENDERING CONSISTENCY
# ═════════════════════════════════════════════════════════════════════════════

class TestUTF8RenderingConsistency:
    """All output artifacts are valid UTF-8 with consistent encoding."""

    def test_json_output_is_valid_utf8(self, tmp_path):
        rpt = tmp_path / "reports"
        _run_pipeline(tmp_path, reports_dir=rpt)
        raw = (rpt / "runtime_summary" / "latest.json").read_bytes()
        raw.decode("utf-8")  # must not raise

    def test_md_output_is_valid_utf8(self, tmp_path):
        rpt = tmp_path / "reports"
        _run_pipeline(tmp_path, reports_dir=rpt)
        raw = (rpt / "runtime_summary" / "latest.md").read_bytes()
        raw.decode("utf-8")

    def test_execution_health_is_valid_utf8(self, tmp_path):
        rpt = tmp_path / "reports"
        _run_pipeline(tmp_path, reports_dir=rpt)
        raw = (rpt / "execution_health" / "latest.json").read_bytes()
        raw.decode("utf-8")

    def test_governance_summary_is_valid_utf8(self, tmp_path):
        rpt = tmp_path / "reports"
        _run_pipeline(tmp_path, reports_dir=rpt)
        raw = (rpt / "governance_summary" / "latest.md").read_bytes()
        raw.decode("utf-8")

    def test_yen_symbol_survives_utf8_roundtrip(self, tmp_path):
        _write_json(tmp_path / "portfolio_state.json", {
            "total_exposure": 1_500_000.0, "drawdown": 0.01,
        })
        decisions = _base_decisions()
        _write_jsonl(tmp_path / "decision_ledger.jsonl", decisions)
        paths = _make_paths(tmp_path)
        rpt   = tmp_path / "reports"
        pipeline = RuntimeSummaryPipeline(
            telemetry_paths=paths, reports_dir=rpt, now_ts=1_700_000_120.0
        )
        pipeline.run()
        md = (rpt / "runtime_summary" / "latest.md").read_text(encoding="utf-8")
        assert "¥" in md

    def test_japanese_symbols_in_missing_data_survive_utf8(self, tmp_path):
        decisions = _base_decisions() + [
            {
                "decision_type": "stale_snapshot",
                "run_id": "run-001",
                "ts": 1_700_000_010.0,
                "metadata": {"missing_symbols": ["3503東証", "7203トヨタ"]},
            }
        ]
        _write_jsonl(tmp_path / "decision_ledger.jsonl", decisions)
        paths = _make_paths(tmp_path)
        rpt   = tmp_path / "reports"
        pipeline = RuntimeSummaryPipeline(
            telemetry_paths=paths, reports_dir=rpt, now_ts=1_700_000_120.0
        )
        pipeline.run()
        md_raw = (rpt / "runtime_summary" / "latest.md").read_bytes()
        md_raw.decode("utf-8")


# ═════════════════════════════════════════════════════════════════════════════
# 4. MISSING TELEMETRY HANDLING
# ═════════════════════════════════════════════════════════════════════════════

class TestMissingTelemetryHandling:
    """When telemetry files are absent, safe defaults must be returned."""

    def test_empty_tmp_produces_safe_defaults(self, tmp_path):
        paths = _make_paths(tmp_path)
        ev    = GovernanceEvaluator(paths=paths, now_ts=1_700_000_120.0)
        s     = ev.evaluate()
        assert s.run.run_status == "unknown"
        assert s.run.run_id     == ""
        assert s.execution.shadow_orders == 0
        assert s.execution.rejected_candidates == 0
        assert s.risk.sector_concentration == 0.0
        assert isinstance(s.risk.emergency_flags, list)

    def test_missing_portfolio_state_safe(self, tmp_path):
        decisions = _base_decisions()
        _write_jsonl(tmp_path / "decision_ledger.jsonl", decisions)
        paths = _make_paths(tmp_path)
        ev    = GovernanceEvaluator(paths=paths, now_ts=1_700_000_120.0)
        s     = ev.evaluate()
        assert s.risk.portfolio_exposure == 0.0
        assert s.risk.drawdown == 0.0
        assert s.risk.sector_concentration == 0.0

    def test_missing_lock_file_safe(self, tmp_path):
        decisions = _base_decisions()
        _write_jsonl(tmp_path / "decision_ledger.jsonl", decisions)
        paths = _make_paths(tmp_path)
        ev    = GovernanceEvaluator(paths=paths, now_ts=1_700_000_120.0)
        s     = ev.evaluate()
        assert s.lock.lock_owner_pid is None
        assert s.lock.heartbeat_gap_sec is None

    def test_missing_drift_metrics_safe(self, tmp_path):
        decisions = _base_decisions()
        _write_jsonl(tmp_path / "decision_ledger.jsonl", decisions)
        paths = _make_paths(tmp_path)
        ev    = GovernanceEvaluator(paths=paths, now_ts=1_700_000_120.0)
        s     = ev.evaluate()
        assert s.data_health.snapshot_latency_p95 is None
        assert s.data_health.cache_reuse_rate == 0.0

    def test_pipeline_runs_with_no_telemetry_files(self, tmp_path):
        paths = _make_paths(tmp_path)
        rpt   = tmp_path / "reports"
        pipeline = RuntimeSummaryPipeline(
            telemetry_paths=paths, reports_dir=rpt, now_ts=1_700_000_120.0
        )
        s = pipeline.run()
        assert s.run.run_status == "unknown"
        assert (rpt / "runtime_summary" / "latest.json").exists()
        assert (rpt / "runtime_summary" / "latest.md").exists()

    def test_deployment_blocked_defaults_true_when_no_telemetry(self, tmp_path):
        paths = _make_paths(tmp_path)
        ev    = GovernanceEvaluator(paths=paths, now_ts=1_700_000_120.0)
        s     = ev.evaluate()
        # run_mode == "unknown" → deployment_blocked=True by default
        assert s.run.deployment_blocked is True

    def test_replay_stable_defaults_true_when_no_divergence_telemetry(self, tmp_path):
        decisions = _base_decisions()
        _write_jsonl(tmp_path / "decision_ledger.jsonl", decisions)
        paths = _make_paths(tmp_path)
        ev    = GovernanceEvaluator(paths=paths, now_ts=1_700_000_120.0)
        s     = ev.evaluate()
        assert s.run.replay_stable is True


# ═════════════════════════════════════════════════════════════════════════════
# 5. RENDERER CONSISTENCY
# ═════════════════════════════════════════════════════════════════════════════

class TestRendererConsistency:
    """JSON and Markdown renderers are consistent with the GovernanceSummary fields."""

    def test_json_contains_all_new_run_fields(self, tmp_path):
        s = _run_pipeline(tmp_path)
        d = JSONRenderer().render_dict(s)
        assert "replay_stable"      in d["run"]
        assert "deployment_blocked" in d["run"]

    def test_json_contains_all_new_execution_fields(self, tmp_path):
        s = _run_pipeline(tmp_path)
        d = JSONRenderer().render_dict(s)
        assert "shadow_orders"       in d["execution"]
        assert "rejected_candidates" in d["execution"]

    def test_json_contains_sector_concentration_in_risk(self, tmp_path):
        s = _run_pipeline(tmp_path)
        d = JSONRenderer().render_dict(s)
        assert "sector_concentration" in d["risk"]

    def test_markdown_contains_deployment_blocked(self, tmp_path):
        s = _run_pipeline(tmp_path)
        s.runtime_score = 90
        md = MarkdownRenderer().render(s)
        assert "deployment_blocked" in md.lower() or "Deployment Blocked" in md

    def test_markdown_contains_shadow_orders(self, tmp_path):
        s = _run_pipeline(tmp_path)
        md = MarkdownRenderer().render(s)
        assert "shadow_orders" in md.lower() or "Shadow Orders" in md

    def test_markdown_contains_replay_stable(self, tmp_path):
        s = _run_pipeline(tmp_path)
        md = MarkdownRenderer().render(s)
        assert "replay" in md.lower()

    def test_markdown_contains_sector_concentration(self, tmp_path):
        s = _run_pipeline(tmp_path)
        md = MarkdownRenderer().render(s)
        assert "sector_concentration" in md.lower() or "Sector" in md

    def test_json_sort_keys_stable(self, tmp_path):
        s  = _run_pipeline(tmp_path)
        jr = JSONRenderer()
        j1 = jr.render(s)
        j2 = jr.render(s)
        assert j1 == j2
        # Keys must be sorted
        data = json.loads(j1)
        top_keys = list(data.keys())
        assert top_keys == sorted(top_keys)

    def test_json_fields_match_dataclass_fields(self, tmp_path):
        s  = _run_pipeline(tmp_path)
        d  = JSONRenderer().render_dict(s)
        # Spot check all sections are present
        for section in ("run", "lock", "universe", "allocation",
                        "execution", "data_health", "governance", "risk", "summary"):
            assert section in d, f"section '{section}' missing from JSON output"

    def test_markdown_execution_section_header_updated(self, tmp_path):
        s  = _run_pipeline(tmp_path)
        md = MarkdownRenderer().render(s)
        assert "Execution Shadow" in md

    def test_markdown_risk_section_has_sector_concentration_row(self, tmp_path):
        _write_json(tmp_path / "portfolio_state.json", {
            "total_exposure": 500_000.0, "drawdown": 0.0,
            "sector_weights": {"IT": 0.35, "Finance": 0.20},
        })
        decisions = _base_decisions()
        _write_jsonl(tmp_path / "decision_ledger.jsonl", decisions)
        paths = _make_paths(tmp_path)
        rpt   = tmp_path / "reports"
        pipeline = RuntimeSummaryPipeline(
            telemetry_paths=paths, reports_dir=rpt, now_ts=1_700_000_120.0
        )
        pipeline.run()
        md = (rpt / "runtime_summary" / "latest.md").read_text(encoding="utf-8")
        assert "sector_concentration" in md


# ═════════════════════════════════════════════════════════════════════════════
# 6. STALE LOCK REPORTING
# ═════════════════════════════════════════════════════════════════════════════

class TestStaleLockReporting:
    """Stale lock events are classified and surfaced correctly."""

    def test_stale_lock_reclaimed_classified_as_warn(self, tmp_path):
        decisions = _base_decisions()
        _write_jsonl(tmp_path / "decision_ledger.jsonl", decisions)
        _write_jsonl(tmp_path / "lock_recovery.jsonl", [
            {"event": "stale_lock_recovered", "stale_mode": "heartbeat_expired", "stale_pid": 9999},
        ])
        paths = _make_paths(tmp_path)
        ev = GovernanceEvaluator(paths=paths, now_ts=1_700_000_120.0)
        cl = SeverityClassifier()
        s  = ev.evaluate()
        a  = cl.classify(s)
        codes = {x.code for x in a}
        assert "STALE_LOCK_RECLAIMED" in codes
        stale_events = [x for x in a if x.code == "STALE_LOCK_RECLAIMED"]
        assert stale_events[0].severity == WARN

    def test_stale_lock_fields_in_lock_section(self, tmp_path):
        decisions = _base_decisions()
        _write_jsonl(tmp_path / "decision_ledger.jsonl", decisions)
        _write_jsonl(tmp_path / "lock_recovery.jsonl", [
            {"event": "stale_lock_recovered", "stale_mode": "pid_dead", "stale_pid": 1234},
        ])
        paths = _make_paths(tmp_path)
        ev = GovernanceEvaluator(paths=paths, now_ts=1_700_000_120.0)
        s  = ev.evaluate()
        assert s.lock.stale_detected is True
        assert s.lock.reclaimed      is True
        assert s.lock.reclaim_reason == "pid_dead"

    def test_heartbeat_gap_triggers_critical_above_120s(self, tmp_path):
        decisions = _base_decisions()
        _write_jsonl(tmp_path / "decision_ledger.jsonl", decisions)
        # Lock with heartbeat 150 seconds ago
        _write_json(tmp_path / "order_lock.json", {
            "pid": 1234, "created_at": 1_699_999_800.0,
            "heartbeat_ts": 1_699_999_970.0, "boot_id": "abc",
        })
        paths = _make_paths(tmp_path)
        ev = GovernanceEvaluator(paths=paths, now_ts=1_700_000_120.0)
        cl = SeverityClassifier()
        s  = ev.evaluate()
        a  = cl.classify(s)
        hb_events = [x for x in a if x.code == "HEARTBEAT_STALE"]
        assert hb_events, "Expected HEARTBEAT_STALE anomaly"
        assert hb_events[0].severity == CRITICAL

    def test_stale_lock_reclaim_reason_in_json_output(self, tmp_path):
        decisions = _base_decisions()
        _write_jsonl(tmp_path / "decision_ledger.jsonl", decisions)
        _write_jsonl(tmp_path / "lock_recovery.jsonl", [
            {"event": "stale_lock_recovered", "stale_mode": "boot_id_changed", "stale_pid": 77},
        ])
        paths = _make_paths(tmp_path)
        ev = GovernanceEvaluator(paths=paths, now_ts=1_700_000_120.0)
        jr = JSONRenderer()
        s  = ev.evaluate()
        d  = jr.render_dict(s)
        assert d["lock"]["stale_detected"] is True
        assert d["lock"]["reclaim_reason"] == "boot_id_changed"

    def test_concurrent_blocked_reported_when_run_skipped(self, tmp_path):
        decisions = _base_decisions() + [
            {
                "decision_type": "run_skipped",
                "run_id": "run-002",
                "ts": 1_700_000_005.0,
                "reason": "overlapping_run in progress",
            }
        ]
        _write_jsonl(tmp_path / "decision_ledger.jsonl", decisions)
        paths = _make_paths(tmp_path)
        ev = GovernanceEvaluator(paths=paths, now_ts=1_700_000_120.0)
        s  = ev.evaluate()
        assert s.lock.concurrent_blocked is True


# ═════════════════════════════════════════════════════════════════════════════
# 7. DIVERGENCE REPORTING
# ═════════════════════════════════════════════════════════════════════════════

class TestDivergenceReporting:
    """Replay divergence is correctly detected and reported."""

    def test_replay_divergence_sets_replay_stable_false(self, tmp_path):
        decisions = _base_decisions()
        _write_jsonl(tmp_path / "decision_ledger.jsonl", decisions)
        _write_jsonl(tmp_path / "circuit_breaker.jsonl", [
            {"event_type": "replay_divergence", "run_id": "run-001", "ts": 1_700_000_050.0},
        ])
        paths = _make_paths(tmp_path)
        ev = GovernanceEvaluator(paths=paths, now_ts=1_700_000_120.0)
        s  = ev.evaluate()
        assert s.governance.replay_match is False
        assert s.run.replay_stable is False

    def test_divergence_score_reflects_divergence_count(self, tmp_path):
        decisions = _base_decisions()
        _write_jsonl(tmp_path / "decision_ledger.jsonl", decisions)
        _write_jsonl(tmp_path / "circuit_breaker.jsonl", [
            {"event_type": "replay_divergence", "run_id": "run-001", "ts": t}
            for t in range(3)
        ])
        paths = _make_paths(tmp_path)
        ev = GovernanceEvaluator(paths=paths, now_ts=1_700_000_120.0)
        s  = ev.evaluate()
        assert s.governance.divergence_score == pytest.approx(0.6, abs=0.01)

    def test_replay_divergence_anomaly_is_critical(self, tmp_path):
        decisions = _base_decisions()
        _write_jsonl(tmp_path / "decision_ledger.jsonl", decisions)
        _write_jsonl(tmp_path / "circuit_breaker.jsonl", [
            {"event_type": "replay_divergence", "run_id": "run-001"},
        ])
        paths = _make_paths(tmp_path)
        ev = GovernanceEvaluator(paths=paths, now_ts=1_700_000_120.0)
        cl = SeverityClassifier()
        s  = ev.evaluate()
        a  = cl.classify(s)
        div = [x for x in a if x.code == "REPLAY_DIVERGENCE"]
        assert div, "Expected REPLAY_DIVERGENCE anomaly"
        assert div[0].severity == CRITICAL

    def test_divergence_caps_score_to_40(self, tmp_path):
        decisions = _base_decisions()
        _write_jsonl(tmp_path / "decision_ledger.jsonl", decisions)
        _write_jsonl(tmp_path / "circuit_breaker.jsonl", [
            {"event_type": "replay_divergence", "run_id": "run-001"},
        ])
        paths = _make_paths(tmp_path)
        ev = GovernanceEvaluator(paths=paths, now_ts=1_700_000_120.0)
        cl = SeverityClassifier()
        sc = RuntimeScoreEngine()
        s  = ev.evaluate()
        a  = cl.classify(s)
        assert sc.compute(s, a) <= 40

    def test_no_divergence_in_clean_run(self, tmp_path):
        decisions = _base_decisions()
        _write_jsonl(tmp_path / "decision_ledger.jsonl", decisions)
        paths = _make_paths(tmp_path)
        ev = GovernanceEvaluator(paths=paths, now_ts=1_700_000_120.0)
        s  = ev.evaluate()
        assert s.run.replay_stable is True
        assert s.governance.divergence_score == 0.0

    def test_broker_mismatch_sets_broker_authoritative_false(self, tmp_path):
        decisions = _base_decisions() + [
            {
                "decision_type": "broker_check_failed",
                "run_id": "run-001",
                "ts": 1_700_000_030.0,
                "reason": "broker mismatch detected in position sync",
            }
        ]
        _write_jsonl(tmp_path / "decision_ledger.jsonl", decisions)
        paths = _make_paths(tmp_path)
        ev = GovernanceEvaluator(paths=paths, now_ts=1_700_000_120.0)
        cl = SeverityClassifier()
        s  = ev.evaluate()
        assert s.governance.broker_authoritative is False
        a  = cl.classify(s)
        codes = {x.code for x in a}
        assert "BROKER_AUTHORITY_MISMATCH" in codes

    def test_divergence_action_appears_in_summary(self, tmp_path):
        decisions = _base_decisions()
        _write_jsonl(tmp_path / "decision_ledger.jsonl", decisions)
        _write_jsonl(tmp_path / "circuit_breaker.jsonl", [
            {"event_type": "replay_divergence", "run_id": "run-001"},
        ])
        paths = _make_paths(tmp_path)
        rpt   = tmp_path / "reports"
        pipeline = RuntimeSummaryPipeline(
            telemetry_paths=paths, reports_dir=rpt, now_ts=1_700_000_120.0
        )
        s = pipeline.run()
        actions_text = " ".join(s.summary.recommended_actions)
        assert "replay" in actions_text.lower()


# ═════════════════════════════════════════════════════════════════════════════
# 8. LATEST SUMMARY POINTER CORRECTNESS
# ═════════════════════════════════════════════════════════════════════════════

class TestLatestSummaryPointerCorrectness:
    """
    reports/runtime_summary/latest.json and latest.md must:
      - exist after pipeline.run()
      - contain the correct run_id from telemetry
      - be valid UTF-8 JSON / Markdown
      - reflect the most-recently-evaluated run when pipeline runs twice
    """

    def test_latest_json_created_after_pipeline_run(self, tmp_path):
        rpt = tmp_path / "reports"
        assert not (rpt / "runtime_summary" / "latest.json").exists()
        _run_pipeline(tmp_path, reports_dir=rpt, run_id="run-abc")
        assert (rpt / "runtime_summary" / "latest.json").exists()

    def test_latest_md_created_after_pipeline_run(self, tmp_path):
        rpt = tmp_path / "reports"
        _run_pipeline(tmp_path, reports_dir=rpt, run_id="run-abc")
        assert (rpt / "runtime_summary" / "latest.md").exists()

    def test_latest_json_run_id_matches_telemetry(self, tmp_path):
        rpt = tmp_path / "reports"
        _run_pipeline(tmp_path, reports_dir=rpt, run_id="run-xyz-99")
        data = json.loads(
            (rpt / "runtime_summary" / "latest.json").read_text(encoding="utf-8")
        )
        assert data["run"]["run_id"] == "run-xyz-99"

    def test_latest_md_contains_run_id(self, tmp_path):
        rpt = tmp_path / "reports"
        _run_pipeline(tmp_path, reports_dir=rpt, run_id="run-test-42")
        md = (rpt / "runtime_summary" / "latest.md").read_text(encoding="utf-8")
        assert "run-test-42" in md

    def test_latest_json_is_parseable(self, tmp_path):
        rpt = tmp_path / "reports"
        _run_pipeline(tmp_path, reports_dir=rpt)
        text = (rpt / "runtime_summary" / "latest.json").read_text(encoding="utf-8")
        data = json.loads(text)
        # Top-level structure
        assert isinstance(data, dict)
        assert "run" in data
        assert "runtime_score" in data

    def test_latest_pointer_updated_on_second_run(self, tmp_path):
        rpt = tmp_path / "reports"
        # First run
        _run_pipeline(tmp_path, reports_dir=rpt, run_id="run-first")
        first_data = json.loads(
            (rpt / "runtime_summary" / "latest.json").read_text(encoding="utf-8")
        )
        assert first_data["run"]["run_id"] == "run-first"

        # Second run with different run_id
        decisions2 = [
            {
                "decision_type": "run_started",
                "run_id": "run-second",
                "epoch_id": "epoch-2",
                "ts": 1_700_001_000.0,
                "metadata": {"mode": "dry"},
            },
            {
                "decision_type": "run_completed",
                "run_id": "run-second",
                "ts": 1_700_001_060.0,
            },
        ]
        # Write new telemetry (overwrite)
        _write_jsonl(tmp_path / "decision_ledger.jsonl", decisions2)
        paths2 = _make_paths(tmp_path)
        pipeline2 = RuntimeSummaryPipeline(
            telemetry_paths=paths2, reports_dir=rpt, now_ts=1_700_001_120.0
        )
        pipeline2.run()

        second_data = json.loads(
            (rpt / "runtime_summary" / "latest.json").read_text(encoding="utf-8")
        )
        assert second_data["run"]["run_id"] == "run-second"

    def test_latest_json_has_runtime_score(self, tmp_path):
        rpt = tmp_path / "reports"
        _run_pipeline(tmp_path, reports_dir=rpt)
        data = json.loads(
            (rpt / "runtime_summary" / "latest.json").read_text(encoding="utf-8")
        )
        assert "runtime_score" in data
        assert 0 <= data["runtime_score"] <= 100

    def test_latest_json_has_all_required_sections(self, tmp_path):
        rpt = tmp_path / "reports"
        _run_pipeline(tmp_path, reports_dir=rpt)
        data = json.loads(
            (rpt / "runtime_summary" / "latest.json").read_text(encoding="utf-8")
        )
        required = {
            "run", "lock", "governance", "execution",
            "data_health", "risk", "summary", "anomalies", "runtime_score",
        }
        missing = required - data.keys()
        assert not missing, f"latest.json missing sections: {missing}"

    def test_latest_json_deployment_blocked_in_run_section(self, tmp_path):
        rpt = tmp_path / "reports"
        _run_pipeline(tmp_path, reports_dir=rpt)
        data = json.loads(
            (rpt / "runtime_summary" / "latest.json").read_text(encoding="utf-8")
        )
        assert "deployment_blocked" in data["run"]
        assert "replay_stable"      in data["run"]

    def test_latest_json_shadow_execution_fields_present(self, tmp_path):
        rpt = tmp_path / "reports"
        _run_pipeline(tmp_path, reports_dir=rpt)
        data = json.loads(
            (rpt / "runtime_summary" / "latest.json").read_text(encoding="utf-8")
        )
        assert "shadow_orders"       in data["execution"]
        assert "rejected_candidates" in data["execution"]

    def test_latest_json_sector_concentration_in_risk(self, tmp_path):
        rpt = tmp_path / "reports"
        _run_pipeline(tmp_path, reports_dir=rpt)
        data = json.loads(
            (rpt / "runtime_summary" / "latest.json").read_text(encoding="utf-8")
        )
        assert "sector_concentration" in data["risk"]

    def test_execution_health_pointer_updated(self, tmp_path):
        rpt = tmp_path / "reports"
        _run_pipeline(tmp_path, reports_dir=rpt, run_id="run-health")
        data = json.loads(
            (rpt / "execution_health" / "latest.json").read_text(encoding="utf-8")
        )
        assert data["run_id"] == "run-health"


# ═════════════════════════════════════════════════════════════════════════════
# Shadow runtime: deployment_blocked + shadow_orders integration
# ═════════════════════════════════════════════════════════════════════════════

class TestShadowRuntimeFields:
    """deployment_blocked and shadow_orders reflect shadow-execution state."""

    def test_deployment_blocked_true_in_dry_mode(self, tmp_path):
        decisions = _base_decisions(mode="dry")
        _write_jsonl(tmp_path / "decision_ledger.jsonl", decisions)
        paths = _make_paths(tmp_path)
        ev = GovernanceEvaluator(paths=paths, now_ts=1_700_000_120.0)
        s  = ev.evaluate()
        assert s.run.deployment_blocked is True

    def test_deployment_blocked_false_in_live_mode(self, tmp_path):
        decisions = _base_decisions(mode="live")
        _write_jsonl(tmp_path / "decision_ledger.jsonl", decisions)
        paths = _make_paths(tmp_path)
        ev = GovernanceEvaluator(paths=paths, now_ts=1_700_000_120.0)
        s  = ev.evaluate()
        # live mode with no explicit deployment-block events → not blocked
        assert s.run.deployment_blocked is False

    def test_shadow_orders_equals_planned_minus_submitted_when_blocked(self, tmp_path):
        # Inject order events: 3 planned, 0 submitted (deployment blocked)
        decisions = _base_decisions(mode="dry") + [
            {"decision_type": "order_planned", "run_id": "run-001", "ts": 1_700_000_010.0},
            {"decision_type": "order_planned", "run_id": "run-001", "ts": 1_700_000_011.0},
            {"decision_type": "order_planned", "run_id": "run-001", "ts": 1_700_000_012.0},
        ]
        _write_jsonl(tmp_path / "decision_ledger.jsonl", decisions)
        paths = _make_paths(tmp_path)
        ev = GovernanceEvaluator(paths=paths, now_ts=1_700_000_120.0)
        s  = ev.evaluate()
        assert s.run.deployment_blocked is True
        # planned_orders should be 3, submitted 0 → shadow_orders 3
        assert s.execution.shadow_orders == s.execution.planned_orders - s.execution.submitted_orders
        assert s.execution.shadow_orders >= 0

    def test_shadow_orders_zero_in_live_mode_with_no_planned(self, tmp_path):
        decisions = _base_decisions(mode="live")
        _write_jsonl(tmp_path / "decision_ledger.jsonl", decisions)
        paths = _make_paths(tmp_path)
        ev = GovernanceEvaluator(paths=paths, now_ts=1_700_000_120.0)
        s  = ev.evaluate()
        # live mode, 0 planned orders → shadow_orders = 0
        assert s.execution.shadow_orders == 0

    def test_rejected_candidates_equals_rejected_allocations(self, tmp_path):
        decisions = _base_decisions()
        _write_jsonl(tmp_path / "decision_ledger.jsonl", decisions)
        _write_jsonl(tmp_path / "deployability.jsonl", [
            {
                "capital_efficiency": 0.6,
                "undeployable_alpha_count": 2,
                "undeployable_count": 4,
                "ts": 1_700_000_010.0,
            }
        ])
        paths = _make_paths(tmp_path)
        ev = GovernanceEvaluator(paths=paths, now_ts=1_700_000_120.0)
        s  = ev.evaluate()
        assert s.execution.rejected_candidates == s.allocation.rejected_allocations
        assert s.execution.rejected_candidates == 4

    def test_sector_concentration_mirrors_allocation_concentration_score(self, tmp_path):
        decisions = _base_decisions() + [
            {
                "decision_type": "alloc_summary",
                "run_id": "run-001",
                "ts": 1_700_000_020.0,
                "metadata": {
                    "sector_utilization": {"IT": 0.45, "Finance": 0.20},
                },
            }
        ]
        _write_jsonl(tmp_path / "decision_ledger.jsonl", decisions)
        paths = _make_paths(tmp_path)
        ev = GovernanceEvaluator(paths=paths, now_ts=1_700_000_120.0)
        s  = ev.evaluate()
        assert s.risk.sector_concentration == pytest.approx(
            s.allocation.concentration_score, abs=1e-6
        )
        assert s.risk.sector_concentration == pytest.approx(0.45, abs=0.001)
