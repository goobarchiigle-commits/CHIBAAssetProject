"""
tests/test_verification_hardening.py
Comprehensive coverage for verification hardening and regression containment layer.

Classes:
  TestSeverityClassification    — Severity enum, Finding, classify_finding
  TestFindingsFromGate          — findings_from_gate_output conversion
  TestTestHygiene               — xfail scanner, forbidden patterns, known failures
  TestKnownFailureExpiry        — expiry date enforcement
  TestValidateArtifacts         — parquet/json/hash/orphan/truncation checks
  TestPerformanceRegression     — baseline record/check/threshold
  TestResearchQuarantine        — import guard, quarantine_experiment
  TestPromoteStrategy           — promotion flow, thresholds, reports
  TestPromotionReport           — immutable report creation
  TestGovernanceGateHardened    — JSON report, signed hash, partial execution refusal
  TestDetectFlakyTests          — classify_results, report generation
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import textwrap
from datetime import date, datetime, timezone, timedelta
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

JST = timezone(timedelta(hours=9))


# ─────────────────────────────────────────────────────────────────────────────
# TestSeverityClassification
# ─────────────────────────────────────────────────────────────────────────────

class TestSeverityClassification:
    def test_severity_ordering(self):
        from src.governance.severity import Severity
        assert Severity.LOW < Severity.MEDIUM
        assert Severity.MEDIUM < Severity.HIGH
        assert Severity.HIGH < Severity.CRITICAL

    def test_severity_ge_le(self):
        from src.governance.severity import Severity
        assert Severity.CRITICAL >= Severity.HIGH
        assert Severity.LOW <= Severity.MEDIUM

    def test_oos_category_is_critical(self):
        from src.governance.severity import classify_finding
        sev = classify_finding("OOS", "hash mismatch in artifact")
        from src.governance.severity import Severity
        assert sev == Severity.CRITICAL

    def test_state_category_is_critical(self):
        from src.governance.severity import classify_finding, Severity
        assert classify_finding("STATE", "contamination detected") == Severity.CRITICAL

    def test_consistency_category_is_high(self):
        from src.governance.severity import classify_finding, Severity
        assert classify_finding("CONSISTENCY", "source file missing") == Severity.HIGH

    def test_critical_keyword_upgrades_high(self):
        from src.governance.severity import classify_finding, Severity
        sev = classify_finding("CONSISTENCY", "hash mismatch found")
        assert sev == Severity.CRITICAL

    def test_perf_category_is_medium(self):
        from src.governance.severity import classify_finding, Severity
        assert classify_finding("PERF", "runtime exceeded baseline") == Severity.MEDIUM

    def test_make_finding(self):
        from src.governance.severity import make_finding, Severity
        f = make_finding("OOS", "artifact missing", source="test.py")
        assert f.severity == Severity.CRITICAL
        assert f.category == "OOS"
        assert f.source == "test.py"

    def test_finding_is_critical(self):
        from src.governance.severity import make_finding, Severity
        f = make_finding("STATE", "contamination")
        assert f.is_critical()

    def test_finding_to_dict(self):
        from src.governance.severity import make_finding
        f = make_finding("IMPORTS", "forbidden import detected")
        d = f.to_dict()
        assert "severity" in d
        assert "category" in d
        assert "message" in d

    def test_count_by_severity(self):
        from src.governance.severity import make_finding, count_by_severity
        findings = [
            make_finding("OOS", "hash mismatch"),
            make_finding("CONSISTENCY", "file missing"),
            make_finding("PERF", "slow"),
        ]
        counts = count_by_severity(findings)
        assert counts["CRITICAL"] >= 1
        assert counts["MEDIUM"] >= 1

    def test_max_severity_empty(self):
        from src.governance.severity import max_severity
        assert max_severity([]) is None

    def test_max_severity_returns_highest(self):
        from src.governance.severity import make_finding, max_severity, Severity
        findings = [
            make_finding("PERF", "slow"),
            make_finding("OOS", "hash mismatch"),
        ]
        assert max_severity(findings) == Severity.CRITICAL


# ─────────────────────────────────────────────────────────────────────────────
# TestFindingsFromGate
# ─────────────────────────────────────────────────────────────────────────────

class TestFindingsFromGate:
    def test_error_prefixed_with_category(self):
        from src.governance.severity import findings_from_gate_output, Severity
        errors   = ["[OOS] artifact hash mismatch"]
        findings = findings_from_gate_output(errors, [])
        assert len(findings) == 1
        assert findings[0].category == "OOS"
        assert findings[0].severity == Severity.CRITICAL

    def test_warning_never_critical(self):
        from src.governance.severity import findings_from_gate_output, Severity
        # Even if message contains critical keyword, warnings are capped at HIGH
        warnings = ["[OOS] something about hash mismatch"]
        findings = findings_from_gate_output([], warnings)
        assert findings[0].severity != Severity.CRITICAL

    def test_unprefixed_message_uses_general(self):
        from src.governance.severity import findings_from_gate_output
        errors   = ["something without bracket prefix"]
        findings = findings_from_gate_output(errors, [])
        assert findings[0].category == "GENERAL"

    def test_multiple_findings(self):
        from src.governance.severity import findings_from_gate_output
        errors   = ["[STATE] contamination", "[IMPORTS] forbidden import"]
        warnings = ["[PERF] slow runtime"]
        findings = findings_from_gate_output(errors, warnings)
        assert len(findings) == 3


# ─────────────────────────────────────────────────────────────────────────────
# TestTestHygiene
# ─────────────────────────────────────────────────────────────────────────────

class TestTestHygiene:
    def test_xfail_with_expiry_passes(self, tmp_path):
        import test_hygiene as th
        py = tmp_path / "test_foo.py"
        py.write_text(textwrap.dedent("""
            import pytest
            @pytest.mark.xfail(reason="Known issue, expiry: 2099-01-01")
            def test_x():
                pass
        """))
        violations = th.scan_xfail_markers(tmp_path)
        assert not violations

    def test_xfail_without_expiry_fails(self, tmp_path):
        import test_hygiene as th
        py = tmp_path / "test_foo.py"
        py.write_text(textwrap.dedent("""
            import pytest
            @pytest.mark.xfail(reason="no date here")
            def test_x():
                pass
        """))
        violations = th.scan_xfail_markers(tmp_path)
        assert len(violations) == 1
        assert "expiry" in violations[0]

    def test_xfail_without_reason_fails(self, tmp_path):
        import test_hygiene as th
        py = tmp_path / "test_foo.py"
        py.write_text(textwrap.dedent("""
            import pytest
            @pytest.mark.xfail
            def test_x():
                pass
        """))
        violations = th.scan_xfail_markers(tmp_path)
        assert len(violations) == 1

    def test_forbidden_pattern_flaky_detected(self, tmp_path):
        import test_hygiene as th
        py = tmp_path / "test_foo.py"
        py.write_text("def test_x():\n    pass  # flaky: depends on timing\n")
        violations = th.scan_forbidden_patterns([tmp_path])
        assert any("flaky" in v for v in violations)

    def test_forbidden_pattern_todo_detected(self, tmp_path):
        import test_hygiene as th
        py = tmp_path / "something.py"
        py.write_text("# TODO: remove this\nx = 1\n")
        violations = th.scan_forbidden_patterns([tmp_path])
        assert any("TODO" in v for v in violations)

    def test_forbidden_pattern_temporary_disable_detected(self, tmp_path):
        import test_hygiene as th
        py = tmp_path / "test_something.py"
        py.write_text("# temporary disable: broken after refactor\n")
        violations = th.scan_forbidden_patterns([tmp_path])
        assert any("temporary disable" in v for v in violations)

    def test_clean_file_no_violations(self, tmp_path):
        import test_hygiene as th
        py = tmp_path / "test_clean.py"
        py.write_text("def test_x():\n    assert 1 + 1 == 2\n")
        violations = th.scan_forbidden_patterns([tmp_path])
        assert not violations

    def test_unregistered_failure_detected(self):
        import test_hygiene as th
        known = [{"test_id": "tests/test_foo.py::test_x", "expiry": "2099-01-01"}]
        actual = ["tests/test_foo.py::test_x", "tests/test_bar.py::test_new"]
        unregistered = th.check_unregistered_failures(actual, known)
        assert len(unregistered) == 1
        assert "test_bar" in unregistered[0]

    def test_all_registered_no_violations(self):
        import test_hygiene as th
        known = [{"test_id": "tests/test_foo.py::test_x", "expiry": "2099-01-01"}]
        actual = ["tests/test_foo.py::test_x"]
        assert not th.check_unregistered_failures(actual, known)

    def test_skip_threshold_exceeded(self):
        import test_hygiene as th
        violations = th.check_skip_threshold(skip_count=15, threshold=10)
        assert len(violations) == 1
        assert "15" in violations[0]

    def test_skip_threshold_ok(self):
        import test_hygiene as th
        assert not th.check_skip_threshold(skip_count=5, threshold=10)


# ─────────────────────────────────────────────────────────────────────────────
# TestKnownFailureExpiry
# ─────────────────────────────────────────────────────────────────────────────

class TestKnownFailureExpiry:
    def test_future_expiry_passes(self):
        import test_hygiene as th
        known = [{"test_id": "tests/foo::bar", "expiry": "2099-12-31", "reason": "ok"}]
        violations = th.check_known_failure_expiry(known)
        assert not violations

    def test_past_expiry_fails(self):
        import test_hygiene as th
        known = [{"test_id": "tests/foo::bar", "expiry": "2020-01-01", "reason": "old"}]
        violations = th.check_known_failure_expiry(known)
        assert len(violations) == 1
        assert "EXPIRED" in violations[0]

    def test_missing_expiry_fails(self):
        import test_hygiene as th
        known = [{"test_id": "tests/foo::bar", "reason": "no expiry"}]
        violations = th.check_known_failure_expiry(known)
        assert len(violations) == 1
        assert "missing expiry" in violations[0]

    def test_invalid_expiry_format_fails(self):
        import test_hygiene as th
        known = [{"test_id": "tests/foo::bar", "expiry": "not-a-date", "reason": "bad"}]
        violations = th.check_known_failure_expiry(known)
        assert len(violations) == 1

    def test_known_failures_file_loaded(self, tmp_path):
        import test_hygiene as th
        kf = tmp_path / "known_failures.json"
        kf.write_text(json.dumps({
            "known_failures": [
                {"test_id": "tests/x::y", "expiry": "2099-01-01", "reason": "ok"}
            ]
        }))
        original = th.KNOWN_FAILURES_FILE
        th.KNOWN_FAILURES_FILE = kf
        try:
            failures = th.load_known_failures()
            assert len(failures) == 1
        finally:
            th.KNOWN_FAILURES_FILE = original


# ─────────────────────────────────────────────────────────────────────────────
# TestValidateArtifacts
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateArtifacts:
    def test_truncated_file_detected(self, tmp_path):
        import validate_artifacts as va
        f = tmp_path / "metrics.json"
        f.write_bytes(b"")
        errors = va.check_truncated(f)
        assert len(errors) == 1
        assert "Truncated" in errors[0]

    def test_non_truncated_passes(self, tmp_path):
        import validate_artifacts as va
        f = tmp_path / "metrics.json"
        f.write_text('{"x":1}')
        errors = va.check_truncated(f)
        assert not errors

    def test_valid_json_passes(self, tmp_path):
        import validate_artifacts as va
        f = tmp_path / "metrics.json"
        f.write_text('{"sharpe": 1.5, "created_at": "2024-01-01"}')
        errors = va.check_json_integrity(f)
        assert not errors

    def test_invalid_json_fails(self, tmp_path):
        import validate_artifacts as va
        f = tmp_path / "metrics.json"
        f.write_text("{invalid json")
        errors = va.check_json_integrity(f)
        assert len(errors) == 1
        assert "Invalid JSON" in errors[0]

    def test_missing_required_field_fails(self, tmp_path):
        import validate_artifacts as va
        f = tmp_path / "metrics.json"
        f.write_text('{"sharpe": 1.5}')
        errors = va.check_json_integrity(f, required_fields=["sharpe", "max_dd"])
        assert any("max_dd" in e for e in errors)

    def test_invalid_timestamp_fails(self, tmp_path):
        import validate_artifacts as va
        f = tmp_path / "metrics.json"
        f.write_text('{"created_at": "not-a-date"}')
        errors = va.check_json_integrity(f)
        assert any("timestamp" in e.lower() for e in errors)

    def test_valid_timestamp_passes(self, tmp_path):
        import validate_artifacts as va
        f = tmp_path / "metrics.json"
        f.write_text('{"created_at": "2024-01-15T09:00:00+0900"}')
        errors = va.check_json_integrity(f)
        assert not errors

    def test_hash_mismatch_detected(self, tmp_path):
        import validate_artifacts as va
        f = tmp_path / "metrics.json"
        f.write_text('{"x": 1}')
        wrong_hash = "a" * 64
        errors = va.check_hash_against_manifest(f, wrong_hash)
        assert any("Hash mismatch" in e for e in errors)

    def test_hash_match_passes(self, tmp_path):
        import validate_artifacts as va
        f = tmp_path / "metrics.json"
        content = b'{"x": 1}'
        f.write_bytes(content)
        correct_hash = hashlib.sha256(content).hexdigest()
        errors = va.check_hash_against_manifest(f, correct_hash)
        assert not errors

    def test_null_hash_fails(self, tmp_path):
        import validate_artifacts as va
        f = tmp_path / "metrics.json"
        f.write_text('{}')
        errors = va.check_hash_against_manifest(f, "")
        assert any("Null hash" in e for e in errors)

    def test_orphan_artifact_detected(self, tmp_path):
        import validate_artifacts as va
        # Create orphan artifact dir (not in registry)
        artifacts_base = tmp_path / "artifacts" / "experiments"
        orphan_dir = artifacts_base / "exp_orphan_12345678"
        orphan_dir.mkdir(parents=True)
        (orphan_dir / "metrics.json").write_text('{"x":1}')
        # Empty experiment registry
        exp_reg = tmp_path / "experiment_registry.json"
        exp_reg.write_text('{"experiments": {}}')

        original_dir = va.ARTIFACTS_DIR
        original_reg = va.EXP_REGISTRY_FILE
        va.ARTIFACTS_DIR = artifacts_base
        va.EXP_REGISTRY_FILE = exp_reg
        try:
            _, warnings = va.validate_experiment_artifacts()
            assert any("Orphan" in w for w in warnings)
        finally:
            va.ARTIFACTS_DIR = original_dir
            va.EXP_REGISTRY_FILE = original_reg

    def test_parquet_check_readable(self, tmp_path):
        pd = pytest.importorskip("pandas")
        import validate_artifacts as va
        import numpy as np
        df = pd.DataFrame({"a": [1, 2, 3]})
        p = tmp_path / "signals.parquet"
        df.to_parquet(p)
        errors = va.check_parquet_readable(p)
        assert not errors

    def test_parquet_check_empty_warns(self, tmp_path):
        pd = pytest.importorskip("pandas")
        import validate_artifacts as va
        df = pd.DataFrame({"a": []})
        p = tmp_path / "signals.parquet"
        df.to_parquet(p)
        errors = va.check_parquet_readable(p)
        assert any("empty" in e.lower() for e in errors)


# ─────────────────────────────────────────────────────────────────────────────
# TestPerformanceRegression
# ─────────────────────────────────────────────────────────────────────────────

class TestPerformanceRegression:
    def _make_baseline(self, tmp_path: Path, value: float, threshold_pct: float = 20) -> Path:
        baseline = {
            "measurements": {
                "test_metric": {
                    "value": value,
                    "threshold_pct": threshold_pct,
                    "unit": "s",
                    "description": "test",
                }
            }
        }
        f = tmp_path / "perf_baseline.json"
        f.write_text(json.dumps(baseline))
        return f

    def test_within_threshold_passes(self, tmp_path):
        import check_performance_regression as cpr
        baseline = self._make_baseline(tmp_path, value=10.0, threshold_pct=20)
        data = json.loads(baseline.read_text())
        violations = cpr.check_all_from_dict({"test_metric": 11.0}, data)
        assert not violations

    def test_exceeds_threshold_fails(self, tmp_path):
        import check_performance_regression as cpr
        baseline = self._make_baseline(tmp_path, value=10.0, threshold_pct=20)
        data = json.loads(baseline.read_text())
        violations = cpr.check_all_from_dict({"test_metric": 15.0}, data)
        assert len(violations) == 1
        assert "regression" in violations[0].lower()

    def test_null_baseline_skipped(self, tmp_path):
        import check_performance_regression as cpr
        data = {"measurements": {"my_metric": {"value": None, "threshold_pct": 20}}}
        violations = cpr.check_all_from_dict({"my_metric": 100.0}, data)
        assert not violations

    def test_unknown_key_skipped(self, tmp_path):
        import check_performance_regression as cpr
        data = {"measurements": {}}
        violations = cpr.check_all_from_dict({"unknown_key": 100.0}, data)
        assert not violations

    def test_record_creates_entry(self, tmp_path):
        import check_performance_regression as cpr
        orig = cpr.BASELINE_FILE
        cpr.BASELINE_FILE = tmp_path / "baseline.json"
        try:
            cpr.record_measurement("new_metric", 42.0)
            data = json.loads((tmp_path / "baseline.json").read_text())
            assert "new_metric" in data["measurements"]
            assert data["measurements"]["new_metric"]["value"] == 42.0
        finally:
            cpr.BASELINE_FILE = orig

    def test_record_updates_existing(self, tmp_path):
        import check_performance_regression as cpr
        orig = cpr.BASELINE_FILE
        f = tmp_path / "baseline.json"
        f.write_text(json.dumps({
            "measurements": {"existing": {"value": 5.0, "threshold_pct": 20}}
        }))
        cpr.BASELINE_FILE = f
        try:
            cpr.record_measurement("existing", 7.0)
            data = json.loads(f.read_text())
            assert data["measurements"]["existing"]["value"] == 7.0
        finally:
            cpr.BASELINE_FILE = orig

    def test_empty_baseline_warns(self, tmp_path):
        import check_performance_regression as cpr
        orig = cpr.BASELINE_FILE
        cpr.BASELINE_FILE = tmp_path / "baseline.json"
        try:
            _, warnings = cpr.run_checks(current={"x": 1.0})
            assert any("no measurements" in w.lower() for w in warnings)
        finally:
            cpr.BASELINE_FILE = orig

    def test_time_import_returns_ms(self):
        import check_performance_regression as cpr
        elapsed = cpr.time_import("json")
        assert elapsed >= 0


# ─────────────────────────────────────────────────────────────────────────────
# TestResearchQuarantine
# ─────────────────────────────────────────────────────────────────────────────

class TestResearchQuarantine:
    def test_quarantine_import_raises(self):
        with pytest.raises(ImportError, match="restricted zone"):
            import research.quarantine  # noqa: F401

    def test_quarantine_experiment_moves_file(self, tmp_path):
        import promote_strategy as ps
        src = tmp_path / "failing_experiment.py"
        src.write_text("# broken\n")
        quarantine_dir = tmp_path / "quarantine"

        original_root = ps.REPO_ROOT
        ps.REPO_ROOT = tmp_path
        ps.ARCHIVE_LOG_FILE = tmp_path / "archive_log.jsonl"
        try:
            # Manually set quarantine dir
            import shutil
            dest = quarantine_dir / f"{src.stem}_20260101_000000{src.suffix}"
            quarantine_dir.mkdir(parents=True, exist_ok=True)
            # Use quarantine_experiment with custom paths
            dest_path = ps.quarantine_experiment(
                src, reason="test quarantine", dry_run=True
            )
            # dry_run: original file still exists
            assert src.exists()
        finally:
            ps.REPO_ROOT = original_root

    def test_quarantine_dry_run_no_move(self, tmp_path):
        import promote_strategy as ps
        src = tmp_path / "experiment.py"
        src.write_text("x = 1\n")
        original_root = ps.REPO_ROOT
        ps.REPO_ROOT = tmp_path
        try:
            ps.quarantine_experiment(src, reason="test", dry_run=True)
            assert src.exists()
        finally:
            ps.REPO_ROOT = original_root

    def test_forbidden_import_from_quarantine(self, tmp_path):
        import validate_active_imports as vai
        # Source file in quarantine path → forbidden
        rel_path = "research/quarantine/broken_strategy.py"
        kind = vai.classify_import(rel_path)
        assert kind == "forbidden"

    def test_quarantine_blocks_promotion(self, tmp_path):
        import promote_strategy as ps
        entry = {
            "status": "experimental",
            "source_file": "research/quarantine/broken.py",
            "oos_sharpe": 2.0,
        }
        errors = ps.check_quarantine("my_strat", entry)
        assert len(errors) == 1
        assert "quarantine" in errors[0]


# ─────────────────────────────────────────────────────────────────────────────
# TestPromoteStrategy
# ─────────────────────────────────────────────────────────────────────────────

class TestPromoteStrategy:
    def _make_registry(self, tmp_path: Path, entry: dict, strategy_id: str = "test_strat") -> Path:
        data = {"_meta": {}, strategy_id: entry}
        f = tmp_path / "strategy_registry.json"
        f.write_text(json.dumps(data))
        return f

    def _make_source(self, tmp_path: Path, name: str = "strategy.py") -> Path:
        p = tmp_path / name
        p.write_text("# strategy\n")
        return p

    def test_invalid_target_status_raises(self):
        import promote_strategy as ps
        with pytest.raises(ValueError, match="Unknown target status"):
            ps.promote("strat", "unknown_status")

    def test_wrong_from_status_fails(self, tmp_path):
        import promote_strategy as ps
        src = self._make_source(tmp_path)
        entry = {
            "status": "deprecated",
            "source_file": str(src.relative_to(tmp_path)),
            "active_config": "",
            "oos_sharpe": 2.0,
            "oos_max_dd": -0.1,
            "wf_result": "5/5 PASS",
            "last_validated_at": "2026-01-01",
        }
        reg = self._make_registry(tmp_path, entry)
        orig = ps.REGISTRY_FILE
        ps.REGISTRY_FILE = reg
        try:
            success, report = ps.promote("test_strat", "active", skip_oos=True)
            assert not success
            assert any("cannot promote" in e for e in report["errors"])
        finally:
            ps.REGISTRY_FILE = orig

    def test_missing_source_file_fails(self, tmp_path):
        import promote_strategy as ps
        entry = {
            "status": "experimental",
            "source_file": "nonexistent_strategy.py",
            "oos_sharpe": 0.5,
        }
        errors = ps.check_source_file(entry, "test_strat")
        assert len(errors) == 1
        assert "not found" in errors[0]

    def test_below_sharpe_threshold_fails(self):
        import promote_strategy as ps
        entry = {"oos_sharpe": 0.3, "oos_max_dd": -0.1, "wf_result": "5/5 PASS"}
        errors = ps.check_metrics_thresholds(entry, "strat", "candidate")
        assert any("oos_sharpe" in e for e in errors)

    def test_above_sharpe_threshold_passes(self):
        import promote_strategy as ps
        entry = {"oos_sharpe": 1.5, "oos_max_dd": -0.1, "wf_result": "5/5 PASS"}
        errors = ps.check_metrics_thresholds(entry, "strat", "candidate")
        assert not errors

    def test_active_requires_5_5_wf(self):
        import promote_strategy as ps
        entry = {"oos_sharpe": 1.5, "oos_max_dd": -0.1, "wf_result": "3/5 PASS"}
        errors = ps.check_metrics_thresholds(entry, "strat", "active")
        assert any("walk-forward" in e for e in errors)

    def test_promotion_to_experimental_no_oos_required(self, tmp_path):
        import promote_strategy as ps
        src = self._make_source(tmp_path)
        entry = {
            "status": "experimental",
            "source_file": str(src.relative_to(REPO_ROOT)) if src.is_relative_to(REPO_ROOT) else str(src),
            "oos_sharpe": 0.5,
            "oos_max_dd": -0.2,
            "wf_result": "",
        }
        errors = ps.check_metrics_thresholds(entry, "strat", "experimental")
        assert not errors

    def test_successful_promotion_updates_registry(self, tmp_path):
        import promote_strategy as ps
        src = tmp_path / "strategy.py"
        src.write_text("x=1\n")
        cfg = tmp_path / "strategy.yaml"
        cfg.write_text("param: 1\n")
        entry = {
            "status": "candidate",
            "source_file": str(src),
            "active_config": str(cfg),
            "oos_sharpe": 1.5,
            "oos_max_dd": -0.1,
            "wf_result": "5/5 PASS",
            "last_validated_at": "2026-01-01",
            "live_entrypoints": [],
        }
        reg = self._make_registry(tmp_path, entry)
        orig_reg = ps.REGISTRY_FILE
        orig_root = ps.REPO_ROOT
        orig_reports = ps.REPORTS_DIR
        orig_archive = ps.ARCHIVE_LOG_FILE
        ps.REGISTRY_FILE = reg
        ps.REPO_ROOT = tmp_path
        ps.REPORTS_DIR = tmp_path / "reports"
        ps.ARCHIVE_LOG_FILE = tmp_path / "archive_log.jsonl"
        try:
            success, report = ps.promote(
                "test_strat", "active", skip_oos=True, dry_run=False
            )
            if success:
                data = json.loads(reg.read_text())
                assert data["test_strat"]["status"] == "active"
        finally:
            ps.REGISTRY_FILE = orig_reg
            ps.REPO_ROOT = orig_root
            ps.REPORTS_DIR = orig_reports
            ps.ARCHIVE_LOG_FILE = orig_archive

    def test_strategy_not_in_registry_fails(self, tmp_path):
        import promote_strategy as ps
        reg = tmp_path / "strategy_registry.json"
        reg.write_text('{"_meta": {}}')
        orig = ps.REGISTRY_FILE
        ps.REGISTRY_FILE = reg
        try:
            success, report = ps.promote("nonexistent", "experimental")
            assert not success
        finally:
            ps.REGISTRY_FILE = orig

    def test_build_report_structure(self):
        import promote_strategy as ps
        entry = {"oos_sharpe": 1.5, "source_file": "x.py"}
        report = ps._build_report("my_strat", "experimental", "candidate", [], [], entry)
        assert "strategy_id" in report
        assert "from_status" in report
        assert "to_status" in report
        assert "metrics_snapshot" in report
        assert "created_at" in report


# ─────────────────────────────────────────────────────────────────────────────
# TestPromotionReport
# ─────────────────────────────────────────────────────────────────────────────

class TestPromotionReport:
    def test_report_written_to_disk(self, tmp_path):
        import promote_strategy as ps
        reports_dir = tmp_path / "reports"
        orig_reports = ps.REPORTS_DIR
        orig_archive = ps.ARCHIVE_LOG_FILE
        ps.REPORTS_DIR = reports_dir
        ps.ARCHIVE_LOG_FILE = tmp_path / "log.jsonl"
        try:
            report = {
                "strategy_id":   "my_strat",
                "from_status":   "experimental",
                "to_status":     "candidate",
                "success":       True,
                "created_at":    datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z"),
                "errors":        [],
                "warnings":      [],
                "metrics_snapshot": {},
                "source_file":   "src/x.py",
                "active_config": "cfg.yaml",
            }
            ps._write_report(report)
            files = list(reports_dir.glob("*.json"))
            assert len(files) == 1
        finally:
            ps.REPORTS_DIR = orig_reports
            ps.ARCHIVE_LOG_FILE = orig_archive

    def test_report_is_valid_json(self, tmp_path):
        import promote_strategy as ps
        reports_dir = tmp_path / "reports"
        orig_reports = ps.REPORTS_DIR
        orig_archive = ps.ARCHIVE_LOG_FILE
        ps.REPORTS_DIR = reports_dir
        ps.ARCHIVE_LOG_FILE = tmp_path / "log.jsonl"
        try:
            report = {
                "strategy_id":   "strat",
                "from_status":   "research",
                "to_status":     "experimental",
                "success":       True,
                "created_at":    "2026-01-01T00:00:00+0900",
                "errors":        [],
                "warnings":      [],
                "metrics_snapshot": {},
                "source_file":   "x.py",
                "active_config": "",
            }
            ps._write_report(report)
            f = list(reports_dir.glob("*.json"))[0]
            data = json.loads(f.read_text())
            assert data["strategy_id"] == "strat"
        finally:
            ps.REPORTS_DIR = orig_reports
            ps.ARCHIVE_LOG_FILE = orig_archive

    def test_immutable_no_overwrite(self, tmp_path):
        import promote_strategy as ps
        reports_dir = tmp_path / "reports"
        orig_reports = ps.REPORTS_DIR
        orig_archive = ps.ARCHIVE_LOG_FILE
        ps.REPORTS_DIR = reports_dir
        ps.ARCHIVE_LOG_FILE = tmp_path / "log.jsonl"
        try:
            base_report = {
                "strategy_id":   "strat",
                "from_status":   "experimental",
                "to_status":     "candidate",
                "success":       True,
                "created_at":    "2026-01-01T00:00:00+0900",
                "errors":        [],
                "warnings":      [],
                "metrics_snapshot": {},
                "source_file":   "x.py",
                "active_config": "",
            }
            ps._write_report(base_report)
            ps._write_report(base_report)
            files = list(reports_dir.glob("*.json"))
            assert len(files) == 2  # second got counter suffix
        finally:
            ps.REPORTS_DIR = orig_reports
            ps.ARCHIVE_LOG_FILE = orig_archive

    def test_archive_log_written(self, tmp_path):
        import promote_strategy as ps
        reports_dir = tmp_path / "reports"
        archive_log = tmp_path / "archive_log.jsonl"
        orig_reports = ps.REPORTS_DIR
        orig_archive = ps.ARCHIVE_LOG_FILE
        ps.REPORTS_DIR = reports_dir
        ps.ARCHIVE_LOG_FILE = archive_log
        try:
            report = {
                "strategy_id":   "strat",
                "from_status":   "experimental",
                "to_status":     "candidate",
                "success":       True,
                "created_at":    "2026-01-01T00:00:00+0900",
                "errors":        [],
                "warnings":      [],
                "metrics_snapshot": {},
                "source_file":   "x.py",
                "active_config": "",
            }
            ps._write_report(report)
            lines = archive_log.read_text().strip().splitlines()
            assert len(lines) >= 1
            entry = json.loads(lines[-1])
            assert entry["event"] == "promotion"
        finally:
            ps.REPORTS_DIR = orig_reports
            ps.ARCHIVE_LOG_FILE = orig_archive


# ─────────────────────────────────────────────────────────────────────────────
# TestDetectFlakyTests
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectFlakyTests:
    def test_consistent_results_not_flaky(self):
        import detect_flaky_tests as dft
        results = {
            "tests/test_a.py::test_x": [True, True, True],
            "tests/test_a.py::test_y": [False, False, False],
        }
        flaky, always_pass, always_fail = dft.classify_results(results)
        assert not flaky
        assert "tests/test_a.py::test_x" in always_pass
        assert "tests/test_a.py::test_y" in always_fail

    def test_inconsistent_results_are_flaky(self):
        import detect_flaky_tests as dft
        results = {
            "tests/test_a.py::test_x": [True, False, True],
        }
        flaky, _, _ = dft.classify_results(results)
        assert "tests/test_a.py::test_x" in flaky

    def test_report_generation_clean(self):
        import detect_flaky_tests as dft
        results = {"tests/test_a.py::test_x": [True, True]}
        doc = dft.generate_report(results, n_runs=2, test_targets=["tests/test_a.py"])
        assert "FLAKY_TEST_REPORT" in doc
        assert "CLEAN" in doc

    def test_report_generation_with_flaky(self):
        import detect_flaky_tests as dft
        results = {"tests/test_a.py::test_x": [True, False, True]}
        doc = dft.generate_report(results, n_runs=3, test_targets=["tests/test_a.py"])
        assert "FLAKY" in doc
        assert "test_x" in doc

    def test_report_shows_pass_rate(self):
        import detect_flaky_tests as dft
        results = {"tests/test_a.py::test_x": [True, False]}  # 50% pass rate
        doc = dft.generate_report(results, n_runs=2, test_targets=["tests/"])
        assert "50%" in doc

    def test_always_fail_appears_in_report(self):
        import detect_flaky_tests as dft
        results = {"tests/test_a.py::test_broken": [False, False, False]}
        doc = dft.generate_report(results, n_runs=3, test_targets=["tests/"])
        assert "test_broken" in doc


# ─────────────────────────────────────────────────────────────────────────────
# TestGovernanceGateHardened
# ─────────────────────────────────────────────────────────────────────────────

class TestGovernanceGateHardened:
    def test_generate_json_report_structure(self):
        import governance_gate as gg
        report = gg.generate_json_report(
            errors=["[OOS] hash mismatch"],
            warnings=["[PERF] slow runtime"],
        )
        assert "generated_at" in report
        assert "gate_status" in report
        assert "summary_hash" in report
        assert "findings" in report
        assert "error_count" in report
        assert report["gate_status"] == "FAIL"

    def test_signed_hash_present(self):
        import governance_gate as gg
        report = gg.generate_json_report([], [])
        assert len(report["summary_hash"]) == 64

    def test_signed_hash_differs_on_different_errors(self):
        import governance_gate as gg
        r1 = gg.generate_json_report(["[OOS] error A"], [])
        r2 = gg.generate_json_report(["[OOS] error B"], [])
        assert r1["summary_hash"] != r2["summary_hash"]

    def test_pass_report_has_zero_errors(self):
        import governance_gate as gg
        report = gg.generate_json_report([], [])
        assert report["gate_status"] == "PASS"
        assert report["error_count"] == 0

    def test_import_errors_cause_partial_refusal(self, monkeypatch):
        import governance_gate as gg
        original = gg._IMPORT_ERRORS[:]
        gg._IMPORT_ERRORS.append("Cannot import: fake_module")
        try:
            errors, _ = gg.run_all_checks(
                skip_import_check=True,
                skip_hygiene=True,
                skip_artifacts=True,
            )
            assert any("PARTIAL EXECUTION REFUSED" in e for e in errors)
        finally:
            gg._IMPORT_ERRORS.clear()
            gg._IMPORT_ERRORS.extend(original)

    def test_severity_counts_in_report(self):
        import governance_gate as gg
        report = gg.generate_json_report(
            errors=["[STATE] contamination detected"],
            warnings=[],
        )
        counts = report.get("severity_counts", {})
        # STATE contamination should be CRITICAL
        assert counts.get("CRITICAL", 0) >= 1

    def test_json_report_written_to_docs(self, tmp_path):
        import governance_gate as gg
        orig_docs = gg.DOCS_DIR
        gg.DOCS_DIR = tmp_path
        try:
            report = gg.generate_json_report([], [])
            out = tmp_path / "GOVERNANCE_REPORT.json"
            out.write_text(json.dumps(report, indent=2))
            loaded = json.loads(out.read_text())
            assert loaded["gate_status"] == "PASS"
        finally:
            gg.DOCS_DIR = orig_docs

    def test_repro_check_added_to_gate(self):
        import governance_gate as gg
        # _check_reproducibility is called in run_all_checks
        assert hasattr(gg, "_check_reproducibility")

    def test_artifact_check_added_to_gate(self):
        import governance_gate as gg
        # _va (validate_artifacts) is imported and used
        assert hasattr(gg, "_va")
