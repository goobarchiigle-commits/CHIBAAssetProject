"""
tests/test_api_auth_diagnostics.py
src/diagnostics/api_auth_diagnostics.py のユニットテスト。

Coverage:
  - record_auth_attempt: JSONL 追記 / フィールド正確性 / fail-open
  - _append_with_rotation: ローテート動作
  - _compute_summary: 7d カウント / 連続失敗 / avg_ms 計算 / empty
  - _load_history: 本体 + ローテートファイル統合
  - get_summary: 不在 / 正常
  - log_auth_diag: [API_AUTH_DIAG] ログ出力
  - log_auth_summary: [API_AUTH_SUMMARY] ログ出力 / no-data
  - _parse_status (startup_check helper)
  - startup_check との後退互換性: _check_api_token_with_retry は (bool,str) を返す
"""
from __future__ import annotations

import json
import logging
import sys
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

JST = timezone(timedelta(hours=9))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_record(
    *,
    api_auth_ok: bool = True,
    response_ms: float = 280.0,
    ts: str | None = None,
    status: object = 200,
) -> dict:
    _ts = ts or datetime.now(JST).isoformat()
    return {
        "ts":                  _ts,
        "status":              status,
        "response_ms":         response_ms,
        "retry_no":            1,
        "retry_max":           6,
        "api_port_ok":         True,
        "api_auth_ok":         api_auth_ok,
        "is_live":             False,
        "mode":                "dry",
        "pw_sha256_8":         "72f9e20f",
        "pw_len":              9,
        "pid":                 12345,
        "user":                "testuser",
        "hostname":            "testhost",
        "startup_elapsed_sec": 0.5,
        "base_url":            "http://localhost:18080/kabusapi",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Test: _append_with_rotation
# ─────────────────────────────────────────────────────────────────────────────

class TestAppendWithRotation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_appends_record(self):
        from src.diagnostics.api_auth_diagnostics import _append_with_rotation
        path = self.base / "test.jsonl"
        rec = {"a": 1}
        _append_with_rotation(path, rec)
        lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0]), {"a": 1})

    def test_appends_multiple(self):
        from src.diagnostics.api_auth_diagnostics import _append_with_rotation
        path = self.base / "test.jsonl"
        for i in range(3):
            _append_with_rotation(path, {"i": i})
        lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 3)

    def test_rotation_on_size_exceeded(self):
        from src.diagnostics import api_auth_diagnostics as mod
        path = self.base / "test.jsonl"
        # Write a tiny file, then patch _MAX_HISTORY_BYTES to 1 to trigger rotation
        _append_with_rotation = mod._append_with_rotation
        path.write_text("existing\n", encoding="utf-8")
        orig = mod._MAX_HISTORY_BYTES
        try:
            mod._MAX_HISTORY_BYTES = 1  # force rotation
            _append_with_rotation(path, {"new": True})
        finally:
            mod._MAX_HISTORY_BYTES = orig
        # original file rotated to .1, new file created
        rotated = path.with_suffix("").with_suffix(".jsonl.1")
        self.assertTrue(rotated.exists(), "rotated file should exist")
        self.assertTrue(path.exists(), "new file should exist")
        self.assertIn('"new": true', path.read_text(encoding="utf-8").lower())


# ─────────────────────────────────────────────────────────────────────────────
# Test: _compute_summary
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeSummary(unittest.TestCase):
    def _ts_ago(self, days: float, hours: float = 0) -> str:
        dt = datetime.now(JST) - timedelta(days=days, hours=hours)
        return dt.isoformat()

    def test_empty_records(self):
        from src.diagnostics.api_auth_diagnostics import _compute_summary
        s = _compute_summary([])
        self.assertEqual(s["success_count_7d"], 0)
        self.assertEqual(s["failure_count_7d"], 0)
        self.assertEqual(s["consecutive_failure_count"], 0)
        self.assertEqual(s["avg_response_ms_success"], 0.0)
        self.assertEqual(s["avg_response_ms_failure"], 0.0)
        self.assertEqual(s["last_success_ts"], "")
        self.assertEqual(s["last_failure_ts"], "")

    def test_all_success_within_7d(self):
        from src.diagnostics.api_auth_diagnostics import _compute_summary
        records = [
            _make_record(api_auth_ok=True, response_ms=200.0, ts=self._ts_ago(1)),
            _make_record(api_auth_ok=True, response_ms=300.0, ts=self._ts_ago(2)),
        ]
        s = _compute_summary(records)
        self.assertEqual(s["success_count_7d"], 2)
        self.assertEqual(s["failure_count_7d"], 0)
        self.assertEqual(s["consecutive_failure_count"], 0)
        self.assertAlmostEqual(s["avg_response_ms_success"], 250.0)

    def test_all_failure_within_7d(self):
        from src.diagnostics.api_auth_diagnostics import _compute_summary
        records = [
            _make_record(api_auth_ok=False, response_ms=276.0, ts=self._ts_ago(1)),
            _make_record(api_auth_ok=False, response_ms=280.0, ts=self._ts_ago(2)),
        ]
        s = _compute_summary(records)
        self.assertEqual(s["success_count_7d"], 0)
        self.assertEqual(s["failure_count_7d"], 2)
        self.assertEqual(s["consecutive_failure_count"], 2)
        self.assertAlmostEqual(s["avg_response_ms_failure"], 278.0)

    def test_7d_cutoff_excludes_old_records(self):
        from src.diagnostics.api_auth_diagnostics import _compute_summary
        records = [
            _make_record(api_auth_ok=True,  response_ms=200.0, ts=self._ts_ago(8)),   # outside 7d
            _make_record(api_auth_ok=False, response_ms=280.0, ts=self._ts_ago(6)),   # inside 7d
        ]
        s = _compute_summary(records)
        self.assertEqual(s["success_count_7d"], 0)  # 8d old excluded
        self.assertEqual(s["failure_count_7d"], 1)
        self.assertNotEqual(s["last_success_ts"], "")  # still tracked

    def test_consecutive_failure_resets_on_success(self):
        from src.diagnostics.api_auth_diagnostics import _compute_summary
        records = [
            _make_record(api_auth_ok=False, ts=self._ts_ago(3)),
            _make_record(api_auth_ok=False, ts=self._ts_ago(2)),
            _make_record(api_auth_ok=True,  ts=self._ts_ago(1)),   # success resets
        ]
        s = _compute_summary(records)
        self.assertEqual(s["consecutive_failure_count"], 0)

    def test_consecutive_failure_counted_from_end(self):
        from src.diagnostics.api_auth_diagnostics import _compute_summary
        records = [
            _make_record(api_auth_ok=True,  ts=self._ts_ago(3)),
            _make_record(api_auth_ok=False, ts=self._ts_ago(2)),
            _make_record(api_auth_ok=False, ts=self._ts_ago(1)),
        ]
        s = _compute_summary(records)
        self.assertEqual(s["consecutive_failure_count"], 2)

    def test_last_ts_tracks_chronologically(self):
        from src.diagnostics.api_auth_diagnostics import _compute_summary
        t1 = self._ts_ago(3)
        t2 = self._ts_ago(1)
        records = [
            _make_record(api_auth_ok=True,  ts=t1),
            _make_record(api_auth_ok=True,  ts=t2),
        ]
        s = _compute_summary(records)
        self.assertEqual(s["last_success_ts"], t2)


# ─────────────────────────────────────────────────────────────────────────────
# Test: record_auth_attempt + get_summary
# ─────────────────────────────────────────────────────────────────────────────

class TestRecordAuthAttempt(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        import src.diagnostics.api_auth_diagnostics as mod
        self._orig_diag_dir    = mod._DIAG_DIR
        self._orig_history     = mod._HISTORY_FILE
        self._orig_summary     = mod._SUMMARY_FILE
        mod._DIAG_DIR     = self.base
        mod._HISTORY_FILE = self.base / "api_auth_history.jsonl"
        mod._SUMMARY_FILE = self.base / "api_auth_summary.json"

    def tearDown(self):
        import src.diagnostics.api_auth_diagnostics as mod
        mod._DIAG_DIR     = self._orig_diag_dir
        mod._HISTORY_FILE = self._orig_history
        mod._SUMMARY_FILE = self._orig_summary
        self.tmp.cleanup()

    def test_jsonl_created_on_first_call(self):
        from src.diagnostics.api_auth_diagnostics import record_auth_attempt, _HISTORY_FILE
        import src.diagnostics.api_auth_diagnostics as mod
        record_auth_attempt(
            status=200, response_ms=281.0, retry_no=1, retry_max=6,
            api_port_ok=True, api_auth_ok=True, is_live=False, mode="dry",
            pw="testpass",
        )
        hist = mod._HISTORY_FILE
        self.assertTrue(hist.exists())
        lines = hist.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)
        r = json.loads(lines[0])
        self.assertEqual(r["api_auth_ok"], True)
        self.assertEqual(r["status"], 200)
        self.assertEqual(r["pw_len"], 8)
        self.assertNotIn("testpass", lines[0])   # plaintext must NOT appear

    def test_multiple_records_appended(self):
        from src.diagnostics.api_auth_diagnostics import record_auth_attempt
        import src.diagnostics.api_auth_diagnostics as mod
        for i in range(3):
            record_auth_attempt(
                status=401, response_ms=276.0, retry_no=i + 1, retry_max=6,
                api_port_ok=True, api_auth_ok=False, is_live=True, mode="live",
            )
        lines = mod._HISTORY_FILE.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 3)

    def test_summary_json_created(self):
        from src.diagnostics.api_auth_diagnostics import record_auth_attempt, get_summary
        import src.diagnostics.api_auth_diagnostics as mod
        record_auth_attempt(
            status=200, response_ms=281.0, retry_no=1, retry_max=6,
            api_port_ok=True, api_auth_ok=True, is_live=False, mode="dry",
        )
        s = mod.get_summary()
        self.assertIn("last_success_ts", s)
        self.assertIn("success_count_7d", s)
        self.assertIn("failure_count_7d", s)
        self.assertIn("consecutive_failure_count", s)
        self.assertIn("avg_response_ms_success", s)
        self.assertIn("avg_response_ms_failure", s)
        self.assertEqual(s["success_count_7d"], 1)

    def test_password_never_in_jsonl(self):
        from src.diagnostics.api_auth_diagnostics import record_auth_attempt
        import src.diagnostics.api_auth_diagnostics as mod
        record_auth_attempt(
            status=401, response_ms=276.0, retry_no=1, retry_max=6,
            api_port_ok=True, api_auth_ok=False, is_live=True, mode="live",
            pw="secret1234",
        )
        content = mod._HISTORY_FILE.read_text(encoding="utf-8")
        self.assertNotIn("secret1234", content)

    def test_fail_open_on_write_error(self):
        from src.diagnostics import api_auth_diagnostics as mod
        from src.diagnostics.api_auth_diagnostics import record_auth_attempt
        # Point files to a non-writable location
        mod._HISTORY_FILE = Path("/nonexistent_dir_abc/api_auth_history.jsonl")
        try:
            record_auth_attempt(
                status=200, response_ms=100.0, retry_no=1, retry_max=1,
                api_port_ok=True, api_auth_ok=True, is_live=False, mode="dry",
            )
        except Exception:
            self.fail("record_auth_attempt must not raise even on write error")


# ─────────────────────────────────────────────────────────────────────────────
# Test: _load_history (with rotated file)
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadHistory(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        import src.diagnostics.api_auth_diagnostics as mod
        self._orig_history = mod._HISTORY_FILE
        mod._HISTORY_FILE  = self.base / "api_auth_history.jsonl"

    def tearDown(self):
        import src.diagnostics.api_auth_diagnostics as mod
        mod._HISTORY_FILE = self._orig_history
        self.tmp.cleanup()

    def test_loads_main_file(self):
        import src.diagnostics.api_auth_diagnostics as mod
        mod._HISTORY_FILE.write_text(
            json.dumps({"ts": "2026-06-13T08:43:00+09:00", "api_auth_ok": True}) + "\n",
            encoding="utf-8",
        )
        records = mod._load_history()
        self.assertEqual(len(records), 1)

    def test_loads_rotated_file(self):
        import src.diagnostics.api_auth_diagnostics as mod
        rotated = mod._HISTORY_FILE.with_suffix("").with_suffix(".jsonl.1")
        rotated.write_text(
            json.dumps({"ts": "2026-06-12T08:43:00+09:00", "api_auth_ok": False}) + "\n",
            encoding="utf-8",
        )
        records = mod._load_history()
        self.assertEqual(len(records), 1)

    def test_loads_both_files(self):
        import src.diagnostics.api_auth_diagnostics as mod
        rotated = mod._HISTORY_FILE.with_suffix("").with_suffix(".jsonl.1")
        mod._HISTORY_FILE.write_text(
            json.dumps({"ts": "2026-06-13T08:43:00+09:00", "api_auth_ok": True}) + "\n",
            encoding="utf-8",
        )
        rotated.write_text(
            json.dumps({"ts": "2026-06-12T08:43:00+09:00", "api_auth_ok": False}) + "\n",
            encoding="utf-8",
        )
        records = mod._load_history()
        self.assertEqual(len(records), 2)

    def test_skips_corrupt_lines(self):
        import src.diagnostics.api_auth_diagnostics as mod
        mod._HISTORY_FILE.write_text(
            "not json\n"
            + json.dumps({"ts": "2026-06-13T08:43:00+09:00", "api_auth_ok": True}) + "\n",
            encoding="utf-8",
        )
        records = mod._load_history()
        self.assertEqual(len(records), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Test: get_summary
# ─────────────────────────────────────────────────────────────────────────────

class TestGetSummary(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        import src.diagnostics.api_auth_diagnostics as mod
        self._orig_summary = mod._SUMMARY_FILE
        mod._SUMMARY_FILE  = Path(self.tmp.name) / "api_auth_summary.json"

    def tearDown(self):
        import src.diagnostics.api_auth_diagnostics as mod
        mod._SUMMARY_FILE = self._orig_summary
        self.tmp.cleanup()

    def test_returns_empty_dict_when_missing(self):
        from src.diagnostics.api_auth_diagnostics import get_summary
        s = get_summary()
        self.assertEqual(s, {})

    def test_returns_valid_dict(self):
        import src.diagnostics.api_auth_diagnostics as mod
        data = {"success_count_7d": 5, "failure_count_7d": 1}
        mod._SUMMARY_FILE.write_text(json.dumps(data), encoding="utf-8")
        s = mod.get_summary()
        self.assertEqual(s["success_count_7d"], 5)

    def test_returns_empty_dict_on_corrupt_json(self):
        import src.diagnostics.api_auth_diagnostics as mod
        mod._SUMMARY_FILE.write_text("not json", encoding="utf-8")
        s = mod.get_summary()
        self.assertEqual(s, {})


# ─────────────────────────────────────────────────────────────────────────────
# Test: log_auth_diag
# ─────────────────────────────────────────────────────────────────────────────

class TestLogAuthDiag(unittest.TestCase):
    def test_outputs_api_auth_diag_prefix(self):
        from src.diagnostics.api_auth_diagnostics import log_auth_diag
        lgr = logging.getLogger("test_diag")
        with self.assertLogs("test_diag", level="WARNING") as cm:
            log_auth_diag(
                lgr,
                retry_no=2, retry_max=6,
                status=401, response_ms=276.5,
                api_port_ok=True, api_auth_ok=False,
                is_live=True, mode="live",
                pw_len=9, pw_sha256_8="72f9e20f",
                startup_elapsed_sec=32.1,
                pid=12345, user="owner", hostname="testpc",
                base_url="http://localhost:18080/kabusapi",
            )
        self.assertTrue(any("[API_AUTH_DIAG]" in m for m in cm.output))

    def test_contains_required_fields(self):
        from src.diagnostics.api_auth_diagnostics import log_auth_diag
        lgr = logging.getLogger("test_diag2")
        with self.assertLogs("test_diag2", level="WARNING") as cm:
            log_auth_diag(
                lgr,
                retry_no=1, retry_max=6, status=401, response_ms=276.0,
                api_port_ok=True, api_auth_ok=False, is_live=True, mode="live",
                pw_len=9, pw_sha256_8="72f9e20f", startup_elapsed_sec=1.0,
                pid=999, user="user1", hostname="host1",
                base_url="http://localhost:18080/kabusapi",
            )
        msg = " ".join(cm.output)
        for field in ["retry=1/6", "status=401", "resp_ms=276", "pw_len=9", "pw_sha256_8=72f9e20f"]:
            self.assertIn(field, msg, f"missing field: {field}")

    def test_never_contains_password_plaintext(self):
        from src.diagnostics.api_auth_diagnostics import log_auth_diag
        lgr = logging.getLogger("test_diag3")
        with self.assertLogs("test_diag3", level="WARNING") as cm:
            log_auth_diag(
                lgr,
                retry_no=1, retry_max=6, status=401, response_ms=276.0,
                api_port_ok=True, api_auth_ok=False, is_live=True, mode="live",
                pw_len=9, pw_sha256_8="72f9e20f", startup_elapsed_sec=1.0,
                pid=999, user="user1", hostname="host1",
                base_url="http://localhost:18080/kabusapi",
            )
        msg = " ".join(cm.output)
        self.assertNotIn("secret", msg)
        self.assertNotIn("password", msg.lower().replace("pw_", "pw"))


# ─────────────────────────────────────────────────────────────────────────────
# Test: log_auth_summary
# ─────────────────────────────────────────────────────────────────────────────

class TestLogAuthSummary(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        import src.diagnostics.api_auth_diagnostics as mod
        self._orig_summary = mod._SUMMARY_FILE
        mod._SUMMARY_FILE  = Path(self.tmp.name) / "api_auth_summary.json"

    def tearDown(self):
        import src.diagnostics.api_auth_diagnostics as mod
        mod._SUMMARY_FILE = self._orig_summary
        self.tmp.cleanup()

    def test_no_data(self):
        from src.diagnostics.api_auth_diagnostics import log_auth_summary
        lgr = logging.getLogger("test_summary_nodata")
        with self.assertLogs("test_summary_nodata", level="INFO") as cm:
            log_auth_summary(lgr)
        self.assertTrue(any("[API_AUTH_SUMMARY]" in m for m in cm.output))
        self.assertTrue(any("no data" in m for m in cm.output))

    def test_with_summary_data(self):
        import src.diagnostics.api_auth_diagnostics as mod
        from src.diagnostics.api_auth_diagnostics import log_auth_summary
        data = {
            "success_count_7d": 24, "failure_count_7d": 2,
            "consecutive_failure_count": 0,
            "last_success_ts": "2026-06-13T08:43:01+09:00",
            "last_failure_ts": "2026-06-12T08:43:01+09:00",
            "avg_response_ms_success": 281.0, "avg_response_ms_failure": 276.0,
            "last_updated": "2026-06-13T08:45:00+09:00",
        }
        mod._SUMMARY_FILE.write_text(json.dumps(data), encoding="utf-8")
        lgr = logging.getLogger("test_summary_data")
        with self.assertLogs("test_summary_data", level="INFO") as cm:
            log_auth_summary(lgr)
        msg = " ".join(cm.output)
        self.assertIn("[API_AUTH_SUMMARY]", msg)
        self.assertIn("7d_success=24", msg)
        self.assertIn("7d_failure=2", msg)
        self.assertIn("consecutive_failures=0", msg)
        self.assertIn("avg_success_ms=281", msg)
        self.assertIn("avg_failure_ms=276", msg)


# ─────────────────────────────────────────────────────────────────────────────
# Test: _parse_status (startup_check helper)
# ─────────────────────────────────────────────────────────────────────────────

class TestParseStatus(unittest.TestCase):
    def test_success_returns_200(self):
        from src.startup_check import _parse_status
        self.assertEqual(_parse_status(True, "API token OK"), 200)

    def test_401_extracted_from_message(self):
        from src.startup_check import _parse_status
        self.assertEqual(
            _parse_status(False, "API認証失敗 (HTTP 401: kabu Station未ログイン)"),
            401,
        )

    def test_503_extracted_from_message(self):
        from src.startup_check import _parse_status
        self.assertEqual(
            _parse_status(False, "API認証失敗 (HTTP 503)"),
            503,
        )

    def test_connection_error(self):
        from src.startup_check import _parse_status
        self.assertEqual(
            _parse_status(False, "API認証失敗 (接続エラー): Connection refused"),
            "ConnectionError",
        )

    def test_timeout(self):
        from src.startup_check import _parse_status
        self.assertEqual(
            _parse_status(False, "API認証失敗 (タイムアウト)"),
            "Timeout",
        )

    def test_value_error(self):
        from src.startup_check import _parse_status
        self.assertEqual(
            _parse_status(False, "API認証不可 (パスワード未設定): password empty"),
            "ValueError",
        )

    def test_unknown_fallback(self):
        from src.startup_check import _parse_status
        self.assertEqual(_parse_status(False, "some unknown error"), "unknown")


# ─────────────────────────────────────────────────────────────────────────────
# Test: _check_api_token_with_retry backward compat (returns bool, str)
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckApiTokenWithRetryCompat(unittest.TestCase):
    def test_returns_bool_and_str_on_success(self):
        from src.startup_check import _check_api_token_with_retry
        with patch("src.startup_check._check_api_token", return_value=(True, "API token OK")):
            with patch("src.startup_check._diag_record_attempt"):
                ok, msg = _check_api_token_with_retry()
        self.assertIsInstance(ok, bool)
        self.assertIsInstance(msg, str)
        self.assertTrue(ok)

    def test_returns_bool_and_str_on_failure(self):
        from src.startup_check import _check_api_token_with_retry
        with patch("src.startup_check._check_api_token", return_value=(False, "API認証失敗 (HTTP 401)")):
            with patch("src.startup_check._retry_sleep"):
                with patch("src.startup_check._diag_record_attempt"):
                    ok, msg = _check_api_token_with_retry(retries=2)
        self.assertIsInstance(ok, bool)
        self.assertIsInstance(msg, str)
        self.assertFalse(ok)

    def test_new_params_have_safe_defaults(self):
        """is_live=False, start_ts=0.0 — should not raise."""
        from src.startup_check import _check_api_token_with_retry
        with patch("src.startup_check._check_api_token", return_value=(True, "API token OK")):
            with patch("src.startup_check._diag_record_attempt"):
                ok, msg = _check_api_token_with_retry(is_live=False, start_ts=0.0)
        self.assertTrue(ok)

    def test_diag_called_once_on_success(self):
        from src.startup_check import _check_api_token_with_retry
        with patch("src.startup_check._check_api_token", return_value=(True, "API token OK")):
            with patch("src.startup_check._diag_record_attempt") as mock_diag:
                _check_api_token_with_retry()
        mock_diag.assert_called_once()

    def test_diag_called_per_attempt_on_failure(self):
        from src.startup_check import _check_api_token_with_retry
        with patch("src.startup_check._check_api_token", return_value=(False, "API認証失敗 (HTTP 401)")):
            with patch("src.startup_check._retry_sleep"):
                with patch("src.startup_check._diag_record_attempt") as mock_diag:
                    _check_api_token_with_retry(retries=3)
        self.assertEqual(mock_diag.call_count, 3)

    def test_diag_fail_open_does_not_affect_result(self):
        """Even if _diag_record_attempt raises, auth result is unaffected."""
        from src.startup_check import _check_api_token_with_retry
        with patch("src.startup_check._check_api_token", return_value=(True, "API token OK")):
            with patch("src.startup_check._diag_record_attempt", side_effect=RuntimeError("diag boom")):
                ok, msg = _check_api_token_with_retry()
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
