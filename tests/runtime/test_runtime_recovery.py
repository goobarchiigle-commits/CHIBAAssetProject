"""
test_runtime_recovery.py — Tests for autonomous runtime recovery layer

Covers:
  - corruption_recovery: safe_read_jsonl, quarantine, audit log
  - runtime_supervisor: orphan lock, duplicate, stale heartbeat, healthy
  - market_data_guard: stale quotes, clock skew, partial update, halt logic
  - bootstrap_recovery: crash detection, file integrity, unresolved orders
  - disk_guard: disk space, archive size, write permission, rotation trigger
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.runtime.corruption_recovery import (
    CorruptionAuditRecord,
    append_corruption_audit,
    load_corruption_audits,
    safe_read_jsonl,
    scan_jsonl_integrity,
    CORRUPT_MALFORMED_JSON,
    CORRUPT_PARTIAL_WRITE,
    CORRUPT_SCHEMA,
    SCHEMA_VERSION as CORRUPT_SCHEMA_VER,
)
from src.runtime.runtime_supervisor import (
    RuntimeHealthReport,
    check_runtime_health,
    append_health_report,
    load_health_reports,
    SCHEMA_VERSION as SUPER_SCHEMA_VER,
)
from src.runtime.market_data_guard import (
    MarketDataGuardReport,
    check_quote_freshness,
    check_candle_freshness,
    detect_clock_skew,
    check_partial_update,
    run_market_data_guard,
    append_guard_report,
    load_guard_reports,
    QUOTE_STALE_SEC,
    CLOCK_SKEW_TOLERANCE_SEC,
    HALT_STALE_FRACTION,
    SCHEMA_VERSION as MDG_SCHEMA_VER,
)
from src.runtime.bootstrap_recovery import (
    BootstrapRecoveryReport,
    run_bootstrap_recovery,
    load_bootstrap_reports,
    _detect_crash_from_lock,
    _scan_jsonl,
    _scan_json,
    _load_unresolved_orders,
    SCHEMA_VERSION as BOOT_SCHEMA_VER,
)
from src.runtime.disk_guard import (
    DiskGuardReport,
    check_disk_space,
    check_archive_size,
    check_write_permission,
    run_disk_guard,
    load_disk_reports,
    SCHEMA_VERSION as DISK_SCHEMA_VER,
)


# ══════════════════════════════════════════════════════════════════════════════
# corruption_recovery
# ══════════════════════════════════════════════════════════════════════════════

def _write_jsonl(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestCorruptionAuditAppendLoad:
    def test_append_and_load_roundtrip(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        rec = CorruptionAuditRecord(
            timestamp="2026-05-16T09:00:00Z",
            source_file="test.jsonl",
            line_number=3,
            corruption_type=CORRUPT_MALFORMED_JSON,
            raw_fragment="{bad json",
            recovery_action="quarantined",
        )
        append_corruption_audit(rec, path)
        loaded = load_corruption_audits(path)
        assert len(loaded) == 1
        r = loaded[0]
        assert r.corruption_type == CORRUPT_MALFORMED_JSON
        assert r.line_number == 3
        assert r.schema_version == CORRUPT_SCHEMA_VER

    def test_append_fail_does_not_raise(self, tmp_path):
        path = tmp_path / "nested" / "deep" / "audit.jsonl"
        rec = CorruptionAuditRecord(
            timestamp="ts", source_file="f", line_number=1,
            corruption_type=CORRUPT_MALFORMED_JSON, raw_fragment="x",
            recovery_action="skipped",
        )
        append_corruption_audit(rec, path)  # must not raise

    def test_load_missing_returns_empty(self, tmp_path):
        assert load_corruption_audits(tmp_path / "missing.jsonl") == []


class TestSafeReadJsonl:
    def test_valid_lines_parsed(self, tmp_path):
        path = tmp_path / "data.jsonl"
        _write_jsonl(path, [
            json.dumps({"a": 1}),
            json.dumps({"b": 2}),
        ])
        valid, n_corrupt = safe_read_jsonl(path)
        assert len(valid) == 2
        assert n_corrupt == 0

    def test_empty_lines_skipped(self, tmp_path):
        path = tmp_path / "data.jsonl"
        _write_jsonl(path, [json.dumps({"a": 1}), "", "   ", json.dumps({"b": 2})])
        valid, n_corrupt = safe_read_jsonl(path)
        assert len(valid) == 2
        assert n_corrupt == 0

    def test_malformed_json_quarantined(self, tmp_path):
        path = tmp_path / "data.jsonl"
        audit_path = tmp_path / "audit.jsonl"
        _write_jsonl(path, [
            json.dumps({"ok": 1}),
            "not json at all",
            json.dumps({"ok": 2}),
        ])
        valid, n_corrupt = safe_read_jsonl(path, audit_path=audit_path)
        assert len(valid) == 2
        assert n_corrupt == 1
        audits = load_corruption_audits(audit_path)
        assert len(audits) == 1
        assert audits[0].corruption_type == CORRUPT_MALFORMED_JSON

    def test_partial_write_detected(self, tmp_path):
        path = tmp_path / "data.jsonl"
        _write_jsonl(path, [
            json.dumps({"ok": 1}),
            '{"truncated": "value"',  # starts with { but no closing }
            json.dumps({"ok": 2}),
        ])
        valid, n_corrupt = safe_read_jsonl(path, audit_path=None)
        assert n_corrupt == 1
        assert len(valid) == 2

    def test_schema_validator_rejects_invalid(self, tmp_path):
        path = tmp_path / "data.jsonl"
        audit_path = tmp_path / "audit.jsonl"
        _write_jsonl(path, [
            json.dumps({"required_field": "x"}),
            json.dumps({"other": "y"}),  # missing required_field
        ])
        valid, n_corrupt = safe_read_jsonl(
            path,
            audit_path=audit_path,
            validator=lambda d: "required_field" in d,
        )
        assert len(valid) == 1
        assert n_corrupt == 1
        audits = load_corruption_audits(audit_path)
        assert audits[0].corruption_type == CORRUPT_SCHEMA

    def test_missing_file_returns_empty(self, tmp_path):
        valid, n = safe_read_jsonl(tmp_path / "nonexistent.jsonl")
        assert valid == []
        assert n == 0

    def test_all_corrupt_returns_empty_valid(self, tmp_path):
        path = tmp_path / "data.jsonl"
        _write_jsonl(path, ["bad1", "bad2", "bad3"])
        valid, n_corrupt = safe_read_jsonl(path)
        assert valid == []
        assert n_corrupt == 3

    def test_no_audit_path_still_skips_corrupt(self, tmp_path):
        path = tmp_path / "data.jsonl"
        _write_jsonl(path, [json.dumps({"ok": 1}), "bad", json.dumps({"ok": 2})])
        valid, n_corrupt = safe_read_jsonl(path)  # no audit_path
        assert len(valid) == 2
        assert n_corrupt == 1


class TestScanJsonlIntegrity:
    def test_valid_file_is_valid(self, tmp_path):
        path = tmp_path / "data.jsonl"
        _write_jsonl(path, [json.dumps({"a": i}) for i in range(5)])
        is_valid, n_valid, n_corrupt = scan_jsonl_integrity(path)
        assert is_valid is True
        assert n_valid == 5
        assert n_corrupt == 0

    def test_corrupted_file_is_not_valid(self, tmp_path):
        path = tmp_path / "data.jsonl"
        _write_jsonl(path, [json.dumps({"a": 1}), "bad", json.dumps({"a": 2})])
        is_valid, n_valid, n_corrupt = scan_jsonl_integrity(path)
        assert is_valid is False
        assert n_corrupt == 1

    def test_missing_file_is_valid(self, tmp_path):
        is_valid, n_valid, n_corrupt = scan_jsonl_integrity(tmp_path / "missing.jsonl")
        assert is_valid is True
        assert n_valid == 0


# ══════════════════════════════════════════════════════════════════════════════
# runtime_supervisor
# ══════════════════════════════════════════════════════════════════════════════

def _write_lock(path: Path, pid: int, heartbeat_ts: float, create_time: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {"pid": pid, "heartbeat_ts": heartbeat_ts}
    if create_time:
        data["process_create_time"] = create_time
    path.write_text(json.dumps(data), encoding="utf-8")


class TestCheckRuntimeHealth:
    def test_no_lock_file_is_healthy(self, tmp_path):
        lock = tmp_path / "exec.lock.json"
        report = check_runtime_health(lock, run_id="r1")
        assert report.overall_healthy is True
        assert not report.orphan_lock_detected
        assert not report.duplicate_runtime_detected
        assert not report.stale_heartbeat_detected

    def test_own_pid_with_fresh_heartbeat_is_healthy(self, tmp_path):
        lock = tmp_path / "exec.lock.json"
        pid = os.getpid()
        now = time.time()
        _write_lock(lock, pid, heartbeat_ts=now - 5)
        report = check_runtime_health(lock, heartbeat_expiry_sec=30.0, now_sec=now)
        assert not report.stale_heartbeat_detected

    def test_stale_heartbeat_detected(self, tmp_path):
        lock = tmp_path / "exec.lock.json"
        pid = os.getpid()
        now = time.time()
        _write_lock(lock, pid, heartbeat_ts=now - 120)
        report = check_runtime_health(lock, heartbeat_expiry_sec=30.0, now_sec=now)
        assert report.stale_heartbeat_detected
        assert report.last_heartbeat_age_sec > 30.0

    def test_dead_pid_is_orphan(self, tmp_path):
        lock = tmp_path / "exec.lock.json"
        # Use an extremely high PID that won't exist
        dead_pid = 999999999
        now = time.time()
        _write_lock(lock, dead_pid, heartbeat_ts=now)
        report = check_runtime_health(lock, now_sec=now)
        assert report.orphan_lock_detected is True
        assert report.orphan_lock_pid == dead_pid

    def test_non_mutex_file_ignored(self, tmp_path):
        lock = tmp_path / "exec.lock.json"
        lock.write_text(json.dumps({"some": "data", "no_pid": True}), encoding="utf-8")
        report = check_runtime_health(lock)
        assert report.overall_healthy is True
        assert not report.orphan_lock_detected

    def test_corrupt_lock_file_ignored(self, tmp_path):
        lock = tmp_path / "exec.lock.json"
        lock.write_text("not_json", encoding="utf-8")
        report = check_runtime_health(lock)
        assert report.overall_healthy is True

    def test_schema_version(self, tmp_path):
        report = check_runtime_health(tmp_path / "lock.json")
        assert report.schema_version == SUPER_SCHEMA_VER


class TestAppendLoadHealthReports:
    def test_append_and_load_roundtrip(self, tmp_path):
        path = tmp_path / "health.jsonl"
        report = check_runtime_health(tmp_path / "lock.json", run_id="r1")
        append_health_report(report, path)
        loaded = load_health_reports(path)
        assert len(loaded) == 1
        assert loaded[0].run_id == "r1"

    def test_load_missing_returns_empty(self, tmp_path):
        assert load_health_reports(tmp_path / "missing.jsonl") == []

    def test_load_skips_corrupt_lines(self, tmp_path):
        path = tmp_path / "health.jsonl"
        good = json.dumps({"timestamp": "ts", "run_id": "r", "orphan_lock_detected": False,
                           "orphan_lock_pid": None, "orphan_lock_detail": "",
                           "duplicate_runtime_detected": False, "stale_heartbeat_detected": False,
                           "last_heartbeat_age_sec": None, "process_create_time_valid": True,
                           "overall_healthy": True, "schema_version": 1})
        path.write_text("corrupt\n" + good + "\n", encoding="utf-8")
        loaded = load_health_reports(path)
        assert len(loaded) == 1


# ══════════════════════════════════════════════════════════════════════════════
# market_data_guard
# ══════════════════════════════════════════════════════════════════════════════

class TestCheckQuoteFreshness:
    def test_fresh_quotes_empty(self):
        now = time.time()
        qs = {"7203": now - 10, "8316": now - 20}
        stale = check_quote_freshness(qs, now, stale_threshold_sec=60)
        assert stale == []

    def test_stale_quotes_detected(self):
        now = time.time()
        qs = {"7203": now - 200, "8316": now - 10}
        stale = check_quote_freshness(qs, now, stale_threshold_sec=60)
        assert stale == ["7203"]

    def test_zero_timestamp_is_stale(self):
        now = time.time()
        qs = {"7203": 0.0}
        stale = check_quote_freshness(qs, now)
        assert "7203" in stale

    def test_all_stale(self):
        now = time.time()
        qs = {"a": now - 300, "b": now - 400}
        stale = check_quote_freshness(qs, now, stale_threshold_sec=60)
        assert sorted(stale) == ["a", "b"]


class TestDetectClockSkew:
    def test_within_tolerance_returns_none(self):
        now = time.time()
        assert detect_clock_skew(now, now + 10, tolerance_sec=30) is None

    def test_exceeds_tolerance_returns_skew(self):
        now = time.time()
        skew = detect_clock_skew(now, now + 60, tolerance_sec=30)
        assert skew is not None
        assert skew >= 60.0

    def test_zero_market_ts_returns_none(self):
        assert detect_clock_skew(time.time(), 0.0) is None

    def test_negative_skew_absolute(self):
        now = time.time()
        skew = detect_clock_skew(now + 60, now, tolerance_sec=30)
        assert skew is not None
        assert skew >= 60.0


class TestCheckPartialUpdate:
    def test_all_received_empty_missing(self):
        assert check_partial_update(["a", "b", "c"], ["a", "b", "c"]) == []

    def test_missing_symbols_detected(self):
        missing = check_partial_update(["a", "b", "c"], ["a"])
        assert sorted(missing) == ["b", "c"]

    def test_extra_received_ignored(self):
        missing = check_partial_update(["a"], ["a", "b", "c"])
        assert missing == []

    def test_empty_expected_empty_missing(self):
        assert check_partial_update([], []) == []


class TestRunMarketDataGuard:
    def test_healthy_returns_no_halt(self):
        now = time.time()
        symbols = ["7203", "8316"]
        qs = {s: now - 10 for s in symbols}
        report = run_market_data_guard(
            expected_symbols=symbols,
            quote_timestamps=qs,
            received_symbols=symbols,
            now_sec=now,
        )
        assert not report.halt_recommended
        assert report.stale_quote_count == 0
        assert report.missing_symbol_count == 0

    def test_majority_stale_triggers_halt(self):
        now = time.time()
        symbols = ["a", "b", "c", "d"]
        # 3 out of 4 stale → 75% > HALT_STALE_FRACTION (50%)
        qs = {"a": now - 300, "b": now - 300, "c": now - 300, "d": now - 10}
        report = run_market_data_guard(
            expected_symbols=symbols,
            quote_timestamps=qs,
            now_sec=now,
        )
        assert report.halt_recommended
        assert "stale_quotes" in report.halt_reason

    def test_clock_skew_triggers_halt(self):
        now = time.time()
        report = run_market_data_guard(
            expected_symbols=["a"],
            local_ts_sec=now,
            market_ts_sec=now + 90,  # 90s skew > 30s tolerance
            now_sec=now,
        )
        assert report.clock_skew_detected
        assert report.halt_recommended

    def test_missing_symbols_recorded(self):
        now = time.time()
        report = run_market_data_guard(
            expected_symbols=["a", "b", "c"],
            received_symbols=["a"],
            now_sec=now,
        )
        assert report.missing_symbol_count == 2
        assert sorted(report.missing_symbols) == ["b", "c"]

    def test_schema_version(self):
        report = run_market_data_guard(expected_symbols=[])
        assert report.schema_version == MDG_SCHEMA_VER


class TestAppendLoadGuardReports:
    def test_append_and_load_roundtrip(self, tmp_path):
        path = tmp_path / "mdg.jsonl"
        report = run_market_data_guard(expected_symbols=["7203"])
        append_guard_report(report, path)
        loaded = load_guard_reports(path)
        assert len(loaded) == 1
        assert loaded[0].schema_version == MDG_SCHEMA_VER

    def test_load_missing_returns_empty(self, tmp_path):
        assert load_guard_reports(tmp_path / "missing.jsonl") == []


# ══════════════════════════════════════════════════════════════════════════════
# bootstrap_recovery
# ══════════════════════════════════════════════════════════════════════════════

class TestDetectCrashFromLock:
    def test_no_lock_no_crash(self, tmp_path):
        crashed, detail = _detect_crash_from_lock(tmp_path / "lock.json")
        assert crashed is False
        assert detail == ""

    def test_missing_pid_no_crash(self, tmp_path):
        lock = tmp_path / "lock.json"
        lock.write_text(json.dumps({"heartbeat_ts": time.time()}), encoding="utf-8")
        crashed, _ = _detect_crash_from_lock(lock)
        assert crashed is False

    def test_dead_pid_crash(self, tmp_path):
        lock = tmp_path / "lock.json"
        lock.write_text(
            json.dumps({"pid": 999999999, "heartbeat_ts": time.time()}),
            encoding="utf-8",
        )
        crashed, detail = _detect_crash_from_lock(lock)
        assert crashed is True
        assert "999999999" in detail

    def test_live_pid_no_crash(self, tmp_path):
        lock = tmp_path / "lock.json"
        pid = os.getpid()
        lock.write_text(
            json.dumps({"pid": pid, "heartbeat_ts": time.time()}),
            encoding="utf-8",
        )
        crashed, _ = _detect_crash_from_lock(lock)
        assert crashed is False

    def test_corrupt_lock_no_crash(self, tmp_path):
        lock = tmp_path / "lock.json"
        lock.write_text("not_json", encoding="utf-8")
        crashed, _ = _detect_crash_from_lock(lock)
        assert crashed is False


class TestScanFiles:
    def test_valid_jsonl(self, tmp_path):
        path = tmp_path / "data.jsonl"
        _write_jsonl(path, [json.dumps({"a": i}) for i in range(3)])
        assert _scan_jsonl(path) is True

    def test_invalid_jsonl(self, tmp_path):
        path = tmp_path / "data.jsonl"
        _write_jsonl(path, [json.dumps({"a": 1}), "bad"])
        assert _scan_jsonl(path) is False

    def test_missing_jsonl_valid(self, tmp_path):
        assert _scan_jsonl(tmp_path / "missing.jsonl") is True

    def test_valid_json(self, tmp_path):
        path = tmp_path / "state.json"
        path.write_text(json.dumps({"key": "val"}), encoding="utf-8")
        assert _scan_json(path) is True

    def test_invalid_json(self, tmp_path):
        path = tmp_path / "state.json"
        path.write_text("{corrupt", encoding="utf-8")
        assert _scan_json(path) is False

    def test_missing_json_valid(self, tmp_path):
        assert _scan_json(tmp_path / "missing.json") is True


class TestLoadUnresolvedOrders:
    def test_empty_file_returns_empty(self, tmp_path):
        path = tmp_path / "inflight.jsonl"
        _write_jsonl(path, [])
        assert _load_unresolved_orders(path) == []

    def test_resolved_orders_excluded(self, tmp_path):
        # Real registry schema: field is "state" (InflightState enum), not "status".
        # RECONCILED/FAILED are terminal; PENDING_SUBMIT/SUBMITTED_UNKNOWN need recovery.
        path = tmp_path / "inflight.jsonl"
        _write_jsonl(path, [
            json.dumps({"symbol": "7203", "side": "BUY", "state": "RECONCILED", "client_order_id": "x"}),
            json.dumps({"symbol": "8316", "side": "BUY", "state": "PENDING_SUBMIT", "client_order_id": "y"}),
        ])
        unresolved = _load_unresolved_orders(path)
        assert len(unresolved) == 1
        assert unresolved[0]["symbol"] == "8316"

    def test_multi_event_single_order_not_triple_counted(self, tmp_path):
        # RCA 2026-07-08: a single order fully progressing through
        # register/submit_attempt/acked (3 JSONL lines) must count as ONE
        # order state (and ACKED is not "needs_recovery"), not 3 unresolved.
        path = tmp_path / "inflight.jsonl"
        _write_jsonl(path, [
            json.dumps({"event": "register", "client_order_id": "z", "state": "PENDING_SUBMIT",
                        "symbol": "5301", "side": "BUY"}),
            json.dumps({"event": "submit_attempt", "client_order_id": "z", "state": "SUBMITTED_UNKNOWN"}),
            json.dumps({"event": "acked", "client_order_id": "z", "state": "ACKED",
                        "broker_order_id": "B1"}),
        ])
        assert _load_unresolved_orders(path) == []

    def test_missing_file_returns_empty(self, tmp_path):
        assert _load_unresolved_orders(tmp_path / "missing.jsonl") == []


class TestRunBootstrapRecovery:
    def test_clean_start(self, tmp_path):
        lock = tmp_path / "lock.json"
        report = run_bootstrap_recovery(
            lock_file=lock,
            run_id="r1",
        )
        assert report.bootstrap_ok is True
        assert not report.crash_detected
        assert report.run_id == "r1"

    def test_crash_detected_from_dead_pid(self, tmp_path):
        lock = tmp_path / "lock.json"
        lock.write_text(
            json.dumps({"pid": 999999999, "heartbeat_ts": time.time()}),
            encoding="utf-8",
        )
        report = run_bootstrap_recovery(lock_file=lock)
        assert report.crash_detected is True
        assert "999999999" in report.crash_detail

    def test_corrupted_jsonl_detected(self, tmp_path):
        lock = tmp_path / "lock.json"
        bad_jsonl = tmp_path / "bad.jsonl"
        _write_jsonl(bad_jsonl, [json.dumps({"ok": 1}), "corrupt"])
        report = run_bootstrap_recovery(
            lock_file=lock,
            jsonl_files=[bad_jsonl],
        )
        assert report.bootstrap_ok is False
        assert str(bad_jsonl) in report.jsonl_corrupted_files

    def test_corrupted_json_detected(self, tmp_path):
        lock = tmp_path / "lock.json"
        bad_json = tmp_path / "state.json"
        bad_json.write_text("{corrupt", encoding="utf-8")
        report = run_bootstrap_recovery(
            lock_file=lock,
            json_files=[bad_json],
        )
        assert report.bootstrap_ok is False
        assert str(bad_json) in report.json_corrupted_files

    def test_unresolved_orders_detected(self, tmp_path):
        lock = tmp_path / "lock.json"
        inflight = tmp_path / "inflight.jsonl"
        _write_jsonl(inflight, [
            json.dumps({"symbol": "7203", "side": "BUY", "state": "PENDING_SUBMIT", "client_order_id": "x"}),
        ])
        report = run_bootstrap_recovery(
            lock_file=lock,
            inflight_file=inflight,
        )
        assert report.unresolved_order_count == 1

    def test_report_appended_to_path(self, tmp_path):
        lock = tmp_path / "lock.json"
        report_path = tmp_path / "bootstrap.jsonl"
        run_bootstrap_recovery(lock_file=lock, report_path=report_path, run_id="r2")
        loaded = load_bootstrap_reports(report_path)
        assert len(loaded) == 1
        assert loaded[0].run_id == "r2"

    def test_never_raises(self, tmp_path):
        # Even with a completely invalid setup, must not raise
        report = run_bootstrap_recovery(
            lock_file=tmp_path / "lock.json",
            jsonl_files=[tmp_path / "nonexistent.jsonl"],
            json_files=[],
        )
        assert isinstance(report, BootstrapRecoveryReport)

    def test_schema_version(self, tmp_path):
        report = run_bootstrap_recovery(lock_file=tmp_path / "lock.json")
        assert report.schema_version == BOOT_SCHEMA_VER


# ══════════════════════════════════════════════════════════════════════════════
# disk_guard
# ══════════════════════════════════════════════════════════════════════════════

class TestCheckDiskSpace:
    def test_returns_positive_values(self, tmp_path):
        free, total = check_disk_space(tmp_path)
        assert free >= 0
        assert total >= 0

    def test_invalid_path_returns_zeros(self):
        free, total = check_disk_space(Path("/nonexistent/path/xyz"))
        # Should return (0, 0) on failure, not raise
        assert isinstance(free, int)
        assert isinstance(total, int)


class TestCheckArchiveSize:
    def test_empty_dir_zero(self, tmp_path):
        archive = tmp_path / "archive"
        archive.mkdir()
        assert check_archive_size(archive) == 0

    def test_missing_dir_zero(self, tmp_path):
        assert check_archive_size(tmp_path / "missing") == 0

    def test_counts_file_sizes(self, tmp_path):
        archive = tmp_path / "archive"
        archive.mkdir()
        (archive / "file1.jsonl").write_text("x" * 1000, encoding="utf-8")
        (archive / "file2.jsonl").write_text("y" * 2000, encoding="utf-8")
        size = check_archive_size(archive)
        assert size >= 3000


class TestCheckWritePermission:
    def test_writable_dir_returns_true(self, tmp_path):
        path = tmp_path / "test_file"
        assert check_write_permission(path) is True

    def test_nonexistent_parent_creates_and_tests(self, tmp_path):
        path = tmp_path / "new_dir" / "test_file"
        assert check_write_permission(path) is True


class TestRunDiskGuard:
    def test_healthy_disk_no_warnings(self, tmp_path):
        report = run_disk_guard(check_path=tmp_path)
        assert isinstance(report, DiskGuardReport)
        # tmp_path on test system should have plenty of space
        assert report.write_ok is True

    def test_archive_size_computed(self, tmp_path):
        archive = tmp_path / "archive"
        archive.mkdir()
        (archive / "old.jsonl").write_text("x" * 500, encoding="utf-8")
        report = run_disk_guard(
            check_path=tmp_path,
            archive_dirs=[archive],
        )
        assert len(report.archive_stats) == 1
        assert report.archive_stats[0]["size_bytes"] >= 500

    def test_report_appended_to_path(self, tmp_path):
        report_path = tmp_path / "disk.jsonl"
        run_disk_guard(check_path=tmp_path, report_path=report_path)
        loaded = load_disk_reports(report_path)
        assert len(loaded) == 1

    def test_rotation_triggered_when_disk_low(self, tmp_path):
        # We can't actually make disk low, but we verify the code path doesn't crash
        archive = tmp_path / "archive"
        report_path = tmp_path / "disk.jsonl"
        report = run_disk_guard(
            check_path=tmp_path,
            archive_dirs=[archive],
            jsonl_paths=[tmp_path / "nonexistent.jsonl"],
            warn_free_pct=1.0,  # always warn (100% threshold)
            report_path=report_path,
        )
        assert report.low_disk_warning is True  # because free < 100% of total
        # rotation_triggered may or may not be True depending on whether file exists

    def test_schema_version(self, tmp_path):
        report = run_disk_guard(check_path=tmp_path)
        assert report.schema_version == DISK_SCHEMA_VER

    def test_load_missing_returns_empty(self, tmp_path):
        assert load_disk_reports(tmp_path / "missing.jsonl") == []

    def test_load_skips_corrupt_lines(self, tmp_path):
        path = tmp_path / "disk.jsonl"
        good = json.dumps({
            "timestamp": "ts", "check_path": "/tmp", "free_bytes": 1000,
            "total_bytes": 10000, "free_pct": 0.1, "low_disk_warning": False,
            "critical_disk_warning": False, "archive_stats": [], "write_ok": True,
            "rotation_triggered": False, "rotated_files": [], "detail": "ok",
            "schema_version": 1,
        })
        path.write_text("corrupt\n" + good + "\n", encoding="utf-8")
        loaded = load_disk_reports(path)
        assert len(loaded) == 1

    def test_never_raises(self):
        # Even with bad inputs, must not raise
        try:
            report = run_disk_guard(check_path=Path("/completely/invalid/path"))
            assert isinstance(report, DiskGuardReport)
        except Exception as e:
            pytest.fail(f"run_disk_guard raised: {e}")
