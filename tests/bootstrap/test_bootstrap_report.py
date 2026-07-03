"""
tests/bootstrap/test_bootstrap_report.py

Coverage:
  - build_bootstrap_report: ABORT on critical failure
  - build_bootstrap_report: CONTINUE on clean bootstrap
  - write_bootstrap_report: atomic write (file exists after call)
  - write_bootstrap_report: tmp file cleaned up
  - report is JSON-serializable
  - operator_summary present with required keys
  - replay compatibility: same inputs → same key outputs
  - initialization_order in report matches toposort
  - fail_fast_reason non-empty
  - run_bootstrap_validation: full pipeline integration
  - run_bootstrap_validation: writes file when output_dir given
  - coherence violation triggers ABORT
"""
import json
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.bootstrap.bootstrap_report import (
    BootstrapReport,
    build_bootstrap_report,
    run_bootstrap_validation,
    write_bootstrap_report,
)
from src.bootstrap.initialization_graph import (
    ALL_PHASES,
    PHASE_ALLOCATION,
    PHASE_BROKER_INTEGRATION,
    PHASE_CAPITAL_STATE,
    PHASE_CONFIGURATION,
    PHASE_GOVERNANCE,
    PHASE_INFLIGHT_REGISTRY,
    PHASE_RECONCILIATION,
    PHASE_RUNTIME_SUPERVISOR,
    RuntimeInitializationGraph,
)

FIXED_TS = datetime(2026, 5, 19, 9, 0, 0, tzinfo=timezone.utc)


class _FakeCapState:
    actual_equity        = 3_000_000.0
    effective_deployable = 2_900_000.0


def _all_init(g: RuntimeInitializationGraph) -> None:
    for p in ALL_PHASES:
        g.mark_initialized(p)


def _full_state():
    return {
        "config":                object(),
        "governance":            object(),
        "capital_state":         _FakeCapState(),
        "allocator":             object(),
        "broker_client":         object(),
        "reconciliation_engine": object(),
        "inflight_registry":     object(),
        "runtime_supervisor":    object(),
        "analytics_registry":    object(),
    }


# ── fail_fast_decision ────────────────────────────────────────────────────────

def test_report_abort_on_critical_phase_failure():
    g = RuntimeInitializationGraph.build_default()
    g.mark_initialized(PHASE_CONFIGURATION)
    g.mark_failed(PHASE_GOVERNANCE, "db unreachable")
    report = build_bootstrap_report(g, run_id="test-001", now_utc=FIXED_TS)
    assert report.fail_fast_decision == "ABORT"


def test_report_continue_on_clean_bootstrap():
    g = RuntimeInitializationGraph.build_default()
    _all_init(g)
    report = build_bootstrap_report(g, run_id="test-002", now_utc=FIXED_TS)
    assert report.fail_fast_decision == "CONTINUE"


def test_report_abort_on_critical_coherence_violation():
    g = RuntimeInitializationGraph.build_default()
    _all_init(g)
    report = build_bootstrap_report(
        g,
        run_id="test-003",
        coherence_violations=["[COHERENCE:CRITICAL] capital_state is None"],
        now_utc=FIXED_TS,
    )
    assert report.fail_fast_decision == "ABORT"


def test_report_continue_on_warn_only_coherence():
    g = RuntimeInitializationGraph.build_default()
    _all_init(g)
    report = build_bootstrap_report(
        g,
        run_id="test-004",
        coherence_violations=["[COHERENCE:WARN] analytics degraded"],
        now_utc=FIXED_TS,
    )
    assert report.fail_fast_decision == "CONTINUE"


# ── content correctness ───────────────────────────────────────────────────────

def test_report_failed_phases_correct():
    g = RuntimeInitializationGraph.build_default()
    g.mark_initialized(PHASE_CONFIGURATION)
    g.mark_failed(PHASE_GOVERNANCE)
    report = build_bootstrap_report(g, run_id="x", now_utc=FIXED_TS)
    assert PHASE_GOVERNANCE in report.failed_phases


def test_report_blocked_phases_correct():
    g = RuntimeInitializationGraph.build_default()
    g.mark_initialized(PHASE_CONFIGURATION)
    g.mark_failed(PHASE_GOVERNANCE)
    report = build_bootstrap_report(g, run_id="x", now_utc=FIXED_TS)
    assert PHASE_CAPITAL_STATE in report.blocked_phases


def test_report_root_cause_summary_non_empty_on_failure():
    g = RuntimeInitializationGraph.build_default()
    g.mark_initialized(PHASE_CONFIGURATION)
    g.mark_failed(PHASE_GOVERNANCE)
    report = build_bootstrap_report(g, run_id="x", now_utc=FIXED_TS)
    assert report.root_cause_summary != ""
    assert PHASE_GOVERNANCE in report.root_cause_summary


def test_report_initialization_order_matches_toposort():
    g = RuntimeInitializationGraph.build_default()
    _all_init(g)
    report = build_bootstrap_report(g, run_id="x", now_utc=FIXED_TS)
    assert report.initialization_order[0] == PHASE_CONFIGURATION
    assert report.initialization_order[-1] == PHASE_RUNTIME_SUPERVISOR


def test_report_fail_fast_reason_non_empty():
    g = RuntimeInitializationGraph.build_default()
    g.mark_initialized(PHASE_CONFIGURATION)
    g.mark_failed(PHASE_GOVERNANCE)
    report = build_bootstrap_report(g, run_id="x", now_utc=FIXED_TS)
    assert len(report.fail_fast_reason) > 0


def test_report_operator_summary_has_required_keys():
    g = RuntimeInitializationGraph.build_default()
    _all_init(g)
    report = build_bootstrap_report(g, run_id="x", now_utc=FIXED_TS)
    assert "ROOT_CAUSE" in report.operator_summary
    assert "BLOCKED_SYSTEMS" in report.operator_summary
    assert "SAFE_ACTION" in report.operator_summary
    assert "REQUIRED_FIX" in report.operator_summary


def test_report_date_matches_now_utc():
    g = RuntimeInitializationGraph.build_default()
    _all_init(g)
    report = build_bootstrap_report(g, run_id="x", now_utc=FIXED_TS)
    assert report.date == "20260519"


def test_report_schema_version_is_one():
    g = RuntimeInitializationGraph.build_default()
    _all_init(g)
    report = build_bootstrap_report(g, run_id="x", now_utc=FIXED_TS)
    assert report.schema_version == 1


# ── JSON serializability ──────────────────────────────────────────────────────

def test_report_is_json_serializable():
    g = RuntimeInitializationGraph.build_default()
    _all_init(g)
    report = build_bootstrap_report(g, run_id="x", now_utc=FIXED_TS)
    data    = asdict(report)
    result  = json.dumps(data, default=str)
    parsed  = json.loads(result)
    assert parsed["schema_version"] == 1


# ── replay compatibility ──────────────────────────────────────────────────────

def test_report_replay_compatible_same_inputs():
    """Two reports from identical graph state and same timestamp must be identical."""
    def _build():
        g = RuntimeInitializationGraph.build_default()
        g.mark_initialized(PHASE_CONFIGURATION)
        g.mark_failed(PHASE_GOVERNANCE)
        return build_bootstrap_report(g, run_id="replay-test", now_utc=FIXED_TS)

    r1 = _build()
    r2 = _build()
    assert r1.fail_fast_decision   == r2.fail_fast_decision
    assert r1.failed_phases        == r2.failed_phases
    assert r1.blocked_phases       == r2.blocked_phases
    assert r1.root_cause_summary   == r2.root_cause_summary
    assert r1.initialization_order == r2.initialization_order


# ── write_bootstrap_report ────────────────────────────────────────────────────

def test_write_bootstrap_report_creates_file(tmp_path):
    g = RuntimeInitializationGraph.build_default()
    _all_init(g)
    report = build_bootstrap_report(g, run_id="write-test", now_utc=FIXED_TS)
    out_dir = tmp_path / "runtime" / "bootstrap"
    path = write_bootstrap_report(report, out_dir)
    assert path.exists()
    assert path.suffix == ".json"


def test_write_bootstrap_report_tmp_cleaned_up(tmp_path):
    g = RuntimeInitializationGraph.build_default()
    _all_init(g)
    report = build_bootstrap_report(g, run_id="write-test-2", now_utc=FIXED_TS)
    out_dir = tmp_path / "bootstrap"
    write_bootstrap_report(report, out_dir)
    tmp_files = list(out_dir.glob("*.tmp"))
    assert tmp_files == [], "tmp file should be deleted after atomic rename"


def test_write_bootstrap_report_valid_json(tmp_path):
    g = RuntimeInitializationGraph.build_default()
    _all_init(g)
    report = build_bootstrap_report(g, run_id="json-test", now_utc=FIXED_TS)
    out_dir = tmp_path / "bootstrap"
    path = write_bootstrap_report(report, out_dir)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["fail_fast_decision"] == "CONTINUE"


def test_write_bootstrap_report_filename_contains_date(tmp_path):
    g = RuntimeInitializationGraph.build_default()
    _all_init(g)
    report = build_bootstrap_report(g, run_id="date-test", now_utc=FIXED_TS)
    out_dir = tmp_path / "bootstrap"
    path = write_bootstrap_report(report, out_dir)
    assert "20260519" in path.name


# ── run_bootstrap_validation ──────────────────────────────────────────────────

def test_run_bootstrap_validation_abort_on_critical_phase_failure():
    g = RuntimeInitializationGraph.build_default()
    g.mark_initialized(PHASE_CONFIGURATION)
    g.mark_failed(PHASE_GOVERNANCE)
    report = run_bootstrap_validation(g, run_id="full-test", now_utc=FIXED_TS)
    assert report.fail_fast_decision == "ABORT"


def test_run_bootstrap_validation_continue_on_full_valid_state():
    g = RuntimeInitializationGraph.build_default()
    _all_init(g)
    report = run_bootstrap_validation(
        g,
        run_id="full-valid",
        state=_full_state(),
        now_utc=FIXED_TS,
    )
    assert report.fail_fast_decision == "CONTINUE"


def test_run_bootstrap_validation_writes_file(tmp_path):
    g = RuntimeInitializationGraph.build_default()
    _all_init(g)
    out_dir = tmp_path / "runtime" / "bootstrap"
    run_bootstrap_validation(
        g,
        run_id="write-full",
        state=_full_state(),
        output_dir=out_dir,
        now_utc=FIXED_TS,
    )
    files = list(out_dir.glob("bootstrap_report_*.json"))
    assert len(files) == 1


def test_run_bootstrap_validation_contract_violations_in_report():
    g = RuntimeInitializationGraph.build_default()
    _all_init(g)
    # capital_state phase is INITIALIZED but no actual cap_state object
    state = dict(_full_state())
    state["capital_state"] = None   # triggers contract violation
    report = run_bootstrap_validation(g, run_id="contract-fail", now_utc=FIXED_TS, state=state)
    # Should have contract violations recorded
    assert len(report.contract_violations) > 0 or len(report.coherence_violations) > 0
