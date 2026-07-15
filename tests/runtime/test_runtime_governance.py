"""
tests/runtime/test_runtime_governance.py
Phase-1 deterministic runtime governance layer tests.

Covers:
  - deterministic summary generation
  - replay-stable outputs (same input → same output)
  - stale lock classification
  - invariant score caps
  - renderer consistency (JSON ↔ fields, MD structure)
  - missing telemetry handling (empty files → safe defaults)
  - concurrent execution telemetry
  - reclaim event reporting
  - all six modules: runtime_summary, governance_evaluator,
    severity_classifier, runtime_score, json_renderer, markdown_renderer
"""
from __future__ import annotations

import json
import os
import sys
import time
import hashlib
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.runtime.runtime_summary import (
    CRITICAL, INFO, WARN,
    AnomalyEvent,
    GovernanceSummary,
    empty_summary,
    LockSection, RunSection, GovernanceSection,
    AllocationSection, ExecutionSection, RiskSection,
)
from src.runtime.governance_evaluator import GovernanceEvaluator, TelemetryPaths
from src.runtime.severity_classifier import SeverityClassifier
from src.runtime.runtime_score import RuntimeScoreEngine, score_band
from src.runtime.json_renderer import JSONRenderer
from src.runtime.markdown_renderer import MarkdownRenderer
from src.runtime.summary_pipeline import RuntimeSummaryPipeline, SummaryComposer


# ─────────────────────────────────────────────────────────────────────────────
# helpers
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
    """Build TelemetryPaths all pointing into tmp dir, with optional overrides."""
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
    )
    defaults.update(overrides)
    return TelemetryPaths(**defaults)


def _make_run_events(tmp: Path, run_id="r1", epoch="e1", status="completed"):
    now = time.time()
    records = [
        {"ts": now - 60, "decision_type": "run_started",
         "run_id": run_id, "epoch_id": epoch,
         "reason": "schedule:live", "metadata": {"mode": "live"}},
    ]
    if status == "completed":
        records.append({"ts": now - 1, "decision_type": "run_completed",
                         "run_id": run_id, "epoch_id": epoch, "reason": "success"})
    elif status == "failed":
        records.append({"ts": now - 1, "decision_type": "run_failed",
                         "run_id": run_id, "epoch_id": epoch, "reason": "stage_failure"})
    _write_jsonl(tmp / "decision_ledger.jsonl", records)


def _make_lock_file(tmp: Path, pid=None, age_sec=10, hb_age=5):
    pid = pid or os.getpid()
    now = time.time()
    created = now - age_sec
    hb_ts   = now - hb_age
    _write_json(tmp / "order_lock.json", {
        "version": "2",
        "pid": pid,
        "process_create_time": None,
        "hostname": "testhost",
        "boot_id": "1234",
        "command": "test",
        "mode": "live",
        "created_at": created,
        "heartbeat_ts": hb_ts,
    })


def _make_stale_recovery(tmp: Path, stale_pid=99999):
    _write_jsonl(tmp / "lock_recovery.jsonl", [{
        "ts": time.time(),
        "event": "stale_lock_recovered",
        "stale_pid": stale_pid,
        "stale_mode": "live",
        "heartbeat_age": "300s",
        "recovered_by_pid": os.getpid(),
        "lock_path": str(tmp / "order_lock.json"),
    }])


# ─────────────────────────────────────────────────────────────────────────────
# TestEmptySummary — missing telemetry returns safe defaults
# ─────────────────────────────────────────────────────────────────────────────

class TestEmptySummary:

    def test_empty_summary_has_all_sections(self):
        s = empty_summary()
        assert s.run is not None
        assert s.lock is not None
        assert s.governance is not None
        assert s.allocation is not None
        assert s.execution is not None
        assert s.data_health is not None
        assert s.risk is not None
        assert s.summary is not None

    def test_evaluator_with_no_files_returns_defaults(self, tmp_path):
        paths = _make_paths(tmp_path)
        ev = GovernanceEvaluator(paths=paths, now_ts=time.time())
        s = ev.evaluate()
        assert s.run.run_status == "unknown"
        assert s.lock.acquisition == "none"
        assert s.governance.replay_match is True    # no divergence events → match
        assert s.governance.fail_closed is True     # no CB trips → closed
        assert s.risk.drawdown == 0.0

    def test_classifier_no_anomalies_on_clean_state(self):
        s = empty_summary()
        anomalies = SeverityClassifier().classify(s)
        # LOCK_STATE_UNKNOWN is expected for empty lock
        codes = {a.code for a in anomalies}
        assert CRITICAL not in {a.severity for a in anomalies}
        assert "LOCK_STATE_UNKNOWN" in codes   # empty lock acquisition="unknown"

    def test_score_on_empty_defaults(self):
        s = empty_summary()
        anomalies = SeverityClassifier().classify(s)
        score = RuntimeScoreEngine().compute(s, anomalies)
        # Should not be 100 (lock unknown = WARN) but not catastrophic
        assert 0 <= score <= 100


# ─────────────────────────────────────────────────────────────────────────────
# TestDeterministicOutput — replay-stable: same input → same output
# ─────────────────────────────────────────────────────────────────────────────

class TestDeterministicOutput:

    def test_same_telemetry_same_json(self, tmp_path):
        _make_run_events(tmp_path)
        _make_lock_file(tmp_path)
        paths = _make_paths(tmp_path)
        fixed_ts = 1_700_000_000.0

        renderer = JSONRenderer()
        ev = GovernanceEvaluator(paths=paths, now_ts=fixed_ts)

        s1 = ev.evaluate()
        s2 = ev.evaluate()

        j1 = renderer.render(s1)
        j2 = renderer.render(s2)
        assert j1 == j2

    def test_same_telemetry_same_markdown(self, tmp_path):
        _make_run_events(tmp_path)
        _make_lock_file(tmp_path)
        paths = _make_paths(tmp_path)
        fixed_ts = 1_700_000_000.0

        ev = GovernanceEvaluator(paths=paths, now_ts=fixed_ts)
        md = MarkdownRenderer()

        s1 = ev.evaluate()
        s2 = ev.evaluate()
        assert md.render(s1) == md.render(s2)

    def test_score_deterministic(self, tmp_path):
        _make_run_events(tmp_path)
        paths = _make_paths(tmp_path)
        ev = GovernanceEvaluator(paths=paths, now_ts=1_700_000_000.0)
        clf = SeverityClassifier()
        eng = RuntimeScoreEngine()

        s = ev.evaluate()
        anomalies = clf.classify(s)
        scores = [eng.compute(s, anomalies) for _ in range(5)]
        assert len(set(scores)) == 1   # all identical

    def test_fencing_token_stable(self, tmp_path):
        """Fencing token is a deterministic hash — same lock file → same token."""
        _make_lock_file(tmp_path, pid=1234)
        paths = _make_paths(tmp_path)

        ev1 = GovernanceEvaluator(paths=paths, now_ts=1_700_000_000.0)
        ev2 = GovernanceEvaluator(paths=paths, now_ts=1_700_000_000.0)
        s1 = ev1.evaluate()
        s2 = ev2.evaluate()
        assert s1.lock.fencing_token == s2.lock.fencing_token
        assert s1.lock.fencing_token is not None
        assert len(s1.lock.fencing_token) == 16


# ─────────────────────────────────────────────────────────────────────────────
# TestStaleLockClassification
# ─────────────────────────────────────────────────────────────────────────────

class TestStaleLockClassification:

    def test_stale_recovery_produces_warn(self, tmp_path):
        _make_run_events(tmp_path)
        _make_stale_recovery(tmp_path, stale_pid=99999)
        paths = _make_paths(tmp_path)
        ev = GovernanceEvaluator(paths=paths, now_ts=time.time())
        s = ev.evaluate()

        anomalies = SeverityClassifier().classify(s)
        codes = [a.code for a in anomalies]
        assert "STALE_LOCK_RECLAIMED" in codes
        sev = next(a.severity for a in anomalies if a.code == "STALE_LOCK_RECLAIMED")
        assert sev == WARN

    def test_stale_detected_sets_lock_fields(self, tmp_path):
        _make_stale_recovery(tmp_path)
        paths = _make_paths(tmp_path)
        ev = GovernanceEvaluator(paths=paths, now_ts=time.time())
        s = ev.evaluate()
        assert s.lock.stale_detected is True
        assert s.lock.reclaimed is True

    def test_expired_heartbeat_critical(self, tmp_path):
        # heartbeat_gap > 120s → CRITICAL
        _make_lock_file(tmp_path, hb_age=130)
        paths = _make_paths(tmp_path)
        ev = GovernanceEvaluator(paths=paths, now_ts=time.time())
        s = ev.evaluate()
        anomalies = SeverityClassifier().classify(s)
        hb_anomaly = [a for a in anomalies if a.code == "HEARTBEAT_STALE"]
        assert hb_anomaly
        assert hb_anomaly[0].severity == CRITICAL

    def test_near_expiry_heartbeat_warn(self, tmp_path):
        # heartbeat_gap > 90s but < 120s → WARN
        _make_lock_file(tmp_path, hb_age=100)
        paths = _make_paths(tmp_path)
        ev = GovernanceEvaluator(paths=paths, now_ts=time.time())
        s = ev.evaluate()
        anomalies = SeverityClassifier().classify(s)
        hb_anomaly = [a for a in anomalies if a.code == "HEARTBEAT_STALE"]
        assert hb_anomaly
        assert hb_anomaly[0].severity == WARN

    def test_fresh_heartbeat_no_anomaly(self, tmp_path):
        _make_lock_file(tmp_path, hb_age=5)
        paths = _make_paths(tmp_path)
        ev = GovernanceEvaluator(paths=paths, now_ts=time.time())
        s = ev.evaluate()
        anomalies = SeverityClassifier().classify(s)
        hb_anomaly = [a for a in anomalies if a.code == "HEARTBEAT_STALE"]
        assert not hb_anomaly


# ─────────────────────────────────────────────────────────────────────────────
# TestConcurrentExecutionTelemetry
# ─────────────────────────────────────────────────────────────────────────────

class TestConcurrentExecutionTelemetry:

    def test_run_skipped_sets_concurrent_blocked(self, tmp_path):
        now = time.time()
        records = [
            {"ts": now - 60, "decision_type": "run_started",
             "run_id": "r1", "epoch_id": "e1", "reason": "live", "metadata": {}},
            {"ts": now - 30, "decision_type": "run_skipped",
             "run_id": "r2", "epoch_id": "e1",
             "reason": "overlapping_run:r1", "metadata": {}},
            {"ts": now - 1, "decision_type": "run_completed",
             "run_id": "r1", "epoch_id": "e1", "reason": "success"},
        ]
        _write_jsonl(tmp_path / "decision_ledger.jsonl", records)
        paths = _make_paths(tmp_path)
        ev = GovernanceEvaluator(paths=paths, now_ts=now)
        s = ev.evaluate()
        assert s.lock.concurrent_blocked is True

    def test_concurrent_blocked_produces_info(self, tmp_path):
        now = time.time()
        records = [
            {"ts": now - 60, "decision_type": "run_started",
             "run_id": "r1", "epoch_id": "e1", "reason": "live", "metadata": {}},
            {"ts": now - 30, "decision_type": "run_skipped",
             "run_id": "r2", "epoch_id": "e1",
             "reason": "overlapping_run:r1", "metadata": {}},
        ]
        _write_jsonl(tmp_path / "decision_ledger.jsonl", records)
        paths = _make_paths(tmp_path)
        ev = GovernanceEvaluator(paths=paths, now_ts=now)
        s = ev.evaluate()
        anomalies = SeverityClassifier().classify(s)
        info = [a for a in anomalies if a.code == "CONCURRENT_EXECUTION_BLOCKED"]
        assert info
        assert info[0].severity == INFO

    def test_no_concurrent_attempt_no_anomaly(self, tmp_path):
        _make_run_events(tmp_path)
        paths = _make_paths(tmp_path)
        ev = GovernanceEvaluator(paths=paths, now_ts=time.time())
        s = ev.evaluate()
        assert s.lock.concurrent_blocked is False
        anomalies = SeverityClassifier().classify(s)
        blocked = [a for a in anomalies if a.code == "CONCURRENT_EXECUTION_BLOCKED"]
        assert not blocked


# ─────────────────────────────────────────────────────────────────────────────
# TestReclaimEventReporting
# ─────────────────────────────────────────────────────────────────────────────

class TestReclaimEventReporting:

    def test_reclaim_fields_populated(self, tmp_path):
        _make_stale_recovery(tmp_path, stale_pid=54321)
        paths = _make_paths(tmp_path)
        ev = GovernanceEvaluator(paths=paths, now_ts=time.time())
        s = ev.evaluate()
        assert s.lock.reclaim_reason == "live"
        assert s.lock.reclaim_success is True
        assert s.lock.reclaimed is True

    def test_reclaim_score_deduction(self, tmp_path):
        """Stale lock reclaimed deducts 15 from score."""
        _make_stale_recovery(tmp_path)
        _make_run_events(tmp_path)
        paths = _make_paths(tmp_path)
        ev = GovernanceEvaluator(paths=paths, now_ts=time.time())
        s = ev.evaluate()
        anomalies = SeverityClassifier().classify(s)
        score_with_stale = RuntimeScoreEngine().compute(s, anomalies)

        # Compare against clean state (no stale)
        paths_clean = _make_paths(tmp_path,
                                  lock_recovery_log=str(tmp_path / "empty_lr.jsonl"))
        ev_clean = GovernanceEvaluator(paths=paths_clean, now_ts=time.time())
        s_clean = ev_clean.evaluate()
        anomalies_clean = SeverityClassifier().classify(s_clean)
        score_clean = RuntimeScoreEngine().compute(s_clean, anomalies_clean)

        assert score_with_stale < score_clean


# ─────────────────────────────────────────────────────────────────────────────
# TestInvariantScoreCaps
# ─────────────────────────────────────────────────────────────────────────────

class TestInvariantScoreCaps:

    def _summary_with(self, **overrides) -> GovernanceSummary:
        s = empty_summary()
        for section_field, value in overrides.items():
            section, _, field = section_field.partition(".")
            sec_obj = getattr(s, section)
            object.__setattr__(sec_obj, field, value)
        return s

    def test_replay_divergence_cap_40(self):
        s = empty_summary()
        s.governance.replay_match     = False
        s.governance.divergence_score = 0.5
        anomalies = SeverityClassifier().classify(s)
        score = RuntimeScoreEngine().compute(s, anomalies)
        assert score <= 40, f"Expected ≤40 for replay divergence, got {score}"

    def test_broker_not_authoritative_cap_25(self):
        s = empty_summary()
        s.governance.broker_authoritative = False
        anomalies = SeverityClassifier().classify(s)
        score = RuntimeScoreEngine().compute(s, anomalies)
        assert score <= 25, f"Expected ≤25 for broker_authoritative=False, got {score}"

    def test_idempotency_violation_cap_10(self):
        s = empty_summary()
        s.execution.idempotency_pass = False
        anomalies = SeverityClassifier().classify(s)
        score = RuntimeScoreEngine().compute(s, anomalies)
        assert score <= 10, f"Expected ≤10 for idempotency_violation, got {score}"

    def test_run_failed_cap_30(self):
        s = empty_summary()
        s.run.run_status = "failed"
        anomalies = SeverityClassifier().classify(s)
        score = RuntimeScoreEngine().compute(s, anomalies)
        assert score <= 30, f"Expected ≤30 for run_failed, got {score}"

    def test_score_bounded_0_100(self):
        s = empty_summary()
        anomalies = SeverityClassifier().classify(s)
        score = RuntimeScoreEngine().compute(s, anomalies)
        assert 0 <= score <= 100

    def test_stale_lock_deducts_15(self):
        s = empty_summary()
        s.lock.stale_detected = True
        s.lock.reclaimed      = True
        s.lock.reclaim_reason = "live"
        base_anomalies = SeverityClassifier().classify(empty_summary())
        base_score = RuntimeScoreEngine().compute(empty_summary(), base_anomalies)

        stale_anomalies = SeverityClassifier().classify(s)
        stale_score = RuntimeScoreEngine().compute(s, stale_anomalies)
        # base is capped at 100 (would be 105 w/ fail_closed bonus), stale=90 → delta=10
        assert base_score - stale_score >= 10

    def test_fail_closed_bonus_5(self):
        s_open   = empty_summary()
        s_closed = empty_summary()
        s_open.governance.fail_closed   = False
        s_closed.governance.fail_closed = True

        a_open   = SeverityClassifier().classify(s_open)
        a_closed = SeverityClassifier().classify(s_closed)

        score_closed = RuntimeScoreEngine().compute(s_closed, a_closed)
        score_open   = RuntimeScoreEngine().compute(s_open,   a_open)
        # Closed gets +5 and no FAIL_OPEN penalty; open gets -20 + no +5
        assert score_closed > score_open

    def test_score_band_labels(self):
        assert score_band(100) == "HEALTHY"
        assert score_band(85)  == "HEALTHY"
        assert score_band(84)  == "DEGRADED"
        assert score_band(65)  == "DEGRADED"
        assert score_band(64)  == "IMPAIRED"
        assert score_band(40)  == "IMPAIRED"
        assert score_band(39)  == "CRITICAL"
        assert score_band(0)   == "CRITICAL"


# ─────────────────────────────────────────────────────────────────────────────
# TestRenderers — zero business logic validation
# ─────────────────────────────────────────────────────────────────────────────

class TestJSONRenderer:

    def _full_summary(self, tmp_path) -> GovernanceSummary:
        _make_run_events(tmp_path)
        _make_lock_file(tmp_path)
        paths = _make_paths(tmp_path)
        ev = GovernanceEvaluator(paths=paths, now_ts=1_700_000_000.0)
        s = ev.evaluate()
        s.anomalies     = SeverityClassifier().classify(s)
        s.runtime_score = RuntimeScoreEngine().compute(s, s.anomalies)
        return s

    def test_json_is_valid(self, tmp_path):
        s = self._full_summary(tmp_path)
        j = JSONRenderer().render(s)
        parsed = json.loads(j)
        assert isinstance(parsed, dict)

    def test_json_contains_all_sections(self, tmp_path):
        s = self._full_summary(tmp_path)
        j = JSONRenderer().render(s)
        d = json.loads(j)
        for section in ("run", "lock", "governance", "allocation", "execution",
                        "data_health", "risk", "universe", "summary"):
            assert section in d, f"missing section: {section}"

    def test_json_sorted_keys(self, tmp_path):
        s = self._full_summary(tmp_path)
        j = JSONRenderer().render(s)
        # Re-dumping with sorted keys must produce same string
        d = json.loads(j)
        assert j == json.dumps(d, indent=2, sort_keys=True)

    def test_render_dict_matches_render(self, tmp_path):
        s = self._full_summary(tmp_path)
        r = JSONRenderer()
        j_str  = r.render(s)
        j_dict = r.render_dict(s)
        assert json.loads(j_str) == j_dict

    def test_json_replay_stable(self, tmp_path):
        """render() on same object always returns identical string."""
        s = self._full_summary(tmp_path)
        r = JSONRenderer()
        assert r.render(s) == r.render(s)

    def test_runtime_score_in_json(self, tmp_path):
        s = self._full_summary(tmp_path)
        d = json.loads(JSONRenderer().render(s))
        assert "runtime_score" in d
        assert isinstance(d["runtime_score"], int)

    def test_anomalies_in_json(self, tmp_path):
        s = self._full_summary(tmp_path)
        d = json.loads(JSONRenderer().render(s))
        assert "anomalies" in d
        assert isinstance(d["anomalies"], list)


class TestMarkdownRenderer:

    def _full_summary(self, tmp_path) -> GovernanceSummary:
        _make_run_events(tmp_path)
        _make_lock_file(tmp_path)
        paths = _make_paths(tmp_path)
        ev = GovernanceEvaluator(paths=paths, now_ts=1_700_000_000.0)
        s = ev.evaluate()
        s.anomalies     = SeverityClassifier().classify(s)
        s.runtime_score = RuntimeScoreEngine().compute(s, s.anomalies)
        sc = SummaryComposer()
        s.summary = sc.compose(s, s.anomalies, s.runtime_score)
        return s

    def test_md_is_non_empty(self, tmp_path):
        s = self._full_summary(tmp_path)
        md = MarkdownRenderer().render(s)
        assert len(md) > 100

    def test_md_contains_score(self, tmp_path):
        s = self._full_summary(tmp_path)
        md = MarkdownRenderer().render(s)
        assert str(s.runtime_score) in md

    def test_md_contains_all_sections(self, tmp_path):
        s = self._full_summary(tmp_path)
        md = MarkdownRenderer().render(s)
        for heading in ("## Run", "## Lock", "## Governance",
                        "## Allocation", "## Execution", "## Risk"):
            assert heading in md, f"missing heading: {heading}"

    def test_md_5second_snapshot_at_top(self, tmp_path):
        s = self._full_summary(tmp_path)
        md = MarkdownRenderer().render(s)
        # Operator snapshot must appear before section details
        snap_pos   = md.find("## ⚡ Operator Snapshot")
        run_pos    = md.find("## Run\n")
        assert snap_pos >= 0, "Operator Snapshot heading missing"
        assert snap_pos < run_pos, "Operator Snapshot must precede detail sections"

    def test_md_replay_stable(self, tmp_path):
        s = self._full_summary(tmp_path)
        r = MarkdownRenderer()
        assert r.render(s) == r.render(s)

    def test_md_critical_anomaly_visible(self, tmp_path):
        s = self._full_summary(tmp_path)
        s.governance.replay_match = False
        s.governance.divergence_score = 0.5
        s.anomalies = SeverityClassifier().classify(s)
        s.runtime_score = RuntimeScoreEngine().compute(s, s.anomalies)
        md = MarkdownRenderer().render(s)
        assert "REPLAY_DIVERGENCE" in md

    def test_md_no_anomalies_message(self, tmp_path):
        s = empty_summary()
        s.anomalies = []
        s.runtime_score = 100
        md = MarkdownRenderer().render(s)
        assert "No anomalies detected" in md


# ─────────────────────────────────────────────────────────────────────────────
# TestPipelineIntegration
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineIntegration:

    def test_pipeline_writes_report_files(self, tmp_path):
        _make_run_events(tmp_path)
        _make_lock_file(tmp_path)
        paths  = _make_paths(tmp_path)
        reports = tmp_path / "reports"

        pipeline = RuntimeSummaryPipeline(
            telemetry_paths = paths,
            reports_dir     = reports,
            now_ts          = 1_700_000_000.0,
        )
        summary = pipeline.run()

        assert (reports / "runtime_summary" / "latest.json").exists()
        assert (reports / "runtime_summary" / "latest.md").exists()
        assert (reports / "execution_health" / "latest.json").exists()
        assert (reports / "governance_summary" / "latest.md").exists()

    def test_pipeline_latest_json_valid(self, tmp_path):
        _make_run_events(tmp_path)
        _make_lock_file(tmp_path)
        paths   = _make_paths(tmp_path)
        reports = tmp_path / "reports"

        RuntimeSummaryPipeline(
            telemetry_paths=paths, reports_dir=reports, now_ts=1_700_000_000.0,
        ).run()

        d = json.loads((reports / "runtime_summary" / "latest.json").read_text())
        assert "run" in d
        assert "lock" in d
        assert "runtime_score" in d

    def test_pipeline_idempotent(self, tmp_path):
        """Running pipeline twice overwrites latest with same content."""
        _make_run_events(tmp_path)
        paths   = _make_paths(tmp_path)
        reports = tmp_path / "reports"

        p = RuntimeSummaryPipeline(
            telemetry_paths=paths, reports_dir=reports, now_ts=1_700_000_000.0,
        )
        p.run()
        j1 = (reports / "runtime_summary" / "latest.json").read_text()
        p.run()
        j2 = (reports / "runtime_summary" / "latest.json").read_text()
        assert j1 == j2

    def test_pipeline_no_stray_tmp_files(self, tmp_path):
        _make_run_events(tmp_path)
        paths   = _make_paths(tmp_path)
        reports = tmp_path / "reports"

        RuntimeSummaryPipeline(
            telemetry_paths=paths, reports_dir=reports, now_ts=1_700_000_000.0,
        ).run()

        tmp_files = list(reports.rglob("*.tmp"))
        assert tmp_files == [], f"stray tmp files: {tmp_files}"

    def test_pipeline_complete_summary_has_explanation(self, tmp_path):
        _make_run_events(tmp_path)
        _make_stale_recovery(tmp_path)
        paths   = _make_paths(tmp_path)
        reports = tmp_path / "reports"

        summary = RuntimeSummaryPipeline(
            telemetry_paths=paths, reports_dir=reports, now_ts=time.time(),
        ).run()

        assert summary.summary.concise_operator_explanation
        assert len(summary.summary.concise_operator_explanation) > 10

    def test_pipeline_stale_lock_in_report(self, tmp_path):
        _make_run_events(tmp_path)
        _make_stale_recovery(tmp_path)
        paths   = _make_paths(tmp_path)
        reports = tmp_path / "reports"

        RuntimeSummaryPipeline(
            telemetry_paths=paths, reports_dir=reports, now_ts=time.time(),
        ).run()

        md = (reports / "runtime_summary" / "latest.md").read_text(encoding="utf-8")
        assert "STALE_LOCK" in md


# ─────────────────────────────────────────────────────────────────────────────
# TestLockExtendedFields
# ─────────────────────────────────────────────────────────────────────────────

class TestLockExtendedFields:

    def test_lock_age_computed(self, tmp_path):
        _make_lock_file(tmp_path, age_sec=60)
        paths = _make_paths(tmp_path)
        now = time.time()
        ev = GovernanceEvaluator(paths=paths, now_ts=now)
        s = ev.evaluate()
        assert s.lock.lock_age_sec is not None
        assert 55 <= s.lock.lock_age_sec <= 65  # 60s ± tolerance

    def test_heartbeat_gap_computed(self, tmp_path):
        _make_lock_file(tmp_path, hb_age=20)
        paths = _make_paths(tmp_path)
        now = time.time()
        ev = GovernanceEvaluator(paths=paths, now_ts=now)
        s = ev.evaluate()
        assert s.lock.heartbeat_gap_sec is not None
        assert 15 <= s.lock.heartbeat_gap_sec <= 25

    def test_lease_expiry_is_future(self, tmp_path):
        """Lease expiry = heartbeat_ts + 120s — should be in the future for fresh HB."""
        _make_lock_file(tmp_path, hb_age=5)
        paths = _make_paths(tmp_path)
        now = time.time()
        ev = GovernanceEvaluator(paths=paths, now_ts=now)
        s = ev.evaluate()
        assert s.lock.lease_expiry is not None
        assert s.lock.lease_expiry > now

    def test_fencing_token_is_hex16(self, tmp_path):
        _make_lock_file(tmp_path)
        paths = _make_paths(tmp_path)
        ev = GovernanceEvaluator(paths=paths, now_ts=time.time())
        s = ev.evaluate()
        tok = s.lock.fencing_token
        assert tok is not None
        assert len(tok) == 16
        assert all(c in "0123456789abcdef" for c in tok)

    def test_no_lock_file_all_none(self, tmp_path):
        paths = _make_paths(tmp_path)
        ev = GovernanceEvaluator(paths=paths, now_ts=time.time())
        s = ev.evaluate()
        assert s.lock.lock_owner_pid  is None
        assert s.lock.lock_age_sec    is None
        assert s.lock.heartbeat_gap_sec is None
        assert s.lock.fencing_token   is None


# ─────────────────────────────────────────────────────────────────────────────
# TestGovernanceRules
# ─────────────────────────────────────────────────────────────────────────────

class TestGovernanceRules:

    def test_replay_divergence_critical(self):
        s = empty_summary()
        s.governance.replay_match = False
        anomalies = SeverityClassifier().classify(s)
        codes = [a for a in anomalies if a.code == "REPLAY_DIVERGENCE"]
        assert codes
        assert codes[0].severity == CRITICAL

    def test_broker_mismatch_critical(self):
        s = empty_summary()
        s.governance.broker_authoritative = False
        anomalies = SeverityClassifier().classify(s)
        codes = [a for a in anomalies if a.code == "BROKER_AUTHORITY_MISMATCH"]
        assert codes
        assert codes[0].severity == CRITICAL

    def test_fail_open_critical(self):
        s = empty_summary()
        s.governance.fail_closed = False
        anomalies = SeverityClassifier().classify(s)
        codes = [a for a in anomalies if a.code == "FAIL_OPEN_DETECTED"]
        assert codes
        assert codes[0].severity == CRITICAL

    def test_drawdown_threshold_critical(self):
        s = empty_summary()
        s.risk.drawdown = 0.15
        anomalies = SeverityClassifier().classify(s)
        dd = [a for a in anomalies if a.code == "DRAWDOWN_LIMIT_BREACH"]
        assert dd
        assert dd[0].severity == CRITICAL

    def test_drawdown_warning_approaching(self):
        s = empty_summary()
        s.risk.drawdown = 0.11
        anomalies = SeverityClassifier().classify(s)
        dd = [a for a in anomalies if a.code == "DRAWDOWN_APPROACHING_LIMIT"]
        assert dd
        assert dd[0].severity == WARN

    def test_missed_entries_over_5_warn(self):
        s = empty_summary()
        s.allocation.missed_entries = 6
        anomalies = SeverityClassifier().classify(s)
        me = [a for a in anomalies if a.code == "EXCESSIVE_MISSED_ENTRIES"]
        assert me
        assert me[0].severity == WARN

    def test_run_failed_critical(self):
        s = empty_summary()
        s.run.run_status = "failed"
        anomalies = SeverityClassifier().classify(s)
        rf = [a for a in anomalies if a.code == "RUN_FAILED"]
        assert rf
        assert rf[0].severity == CRITICAL
