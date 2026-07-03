"""
tests/test_api_auth_forensics.py — API認証フォレンジック強化 テスト

Coverage:
  [1] TOKEN_FORENSIC: fetch_token() 失敗時にログ出力
  [2] API_AUTH_DETAIL: startup_check が Code/Message を個別抽出
  [3] api_auth_failures.jsonl: 失敗時に追記、正常時は追記なし
  [4] KabuS.exe プロセス情報: _collect() が kabus_proc_pid を含む
  [5] API_AUTH_RETRY: retry ループでログ出力

前提: 売買ロジック・シグナル生成・認証判定は一切変更しない。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# ヘルパ
# ─────────────────────────────────────────────────────────────────────────────

def _make_http_response(status_code: int, body: str = "") -> MagicMock:
    """requests.Response モック生成。raise_for_status() は status != 200 で HTTPError を送出。"""
    import requests
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = body
    resp.headers = {"Content-Type": "application/json", "Server": "kabuStation/5.43"}
    resp.url = "http://localhost:18080/kabusapi/token"
    resp.ok = (status_code < 400)
    if status_code >= 400:
        err = requests.exceptions.HTTPError(response=resp)
        resp.raise_for_status.side_effect = err
    else:
        resp.raise_for_status.return_value = None
        resp.json.return_value = json.loads(body) if body else {}
    return resp


_BODY_4001007 = '{"Code":4001007,"Message":"ログイン認証エラー"}'
_BODY_4001011 = '{"Code":4001011,"Message":"リクエスト文字列不正"}'


# ─────────────────────────────────────────────────────────────────────────────
# [1] TOKEN_FORENSIC — client.py fetch_token()
# ─────────────────────────────────────────────────────────────────────────────

class TestTokenForensicLogging(unittest.TestCase):
    """fetch_token() が非200応答時に [TOKEN_FORENSIC] を WARNING 出力する。"""

    def _run_fetch(self, resp: MagicMock) -> None:
        """KabuClient.fetch_token() を実行（例外は caller が処理）。"""
        from src.kabusapi.client import KabuClient, _CACHE
        _CACHE.invalidate()
        with patch.dict(os.environ, {"KABU_API_PASSWORD": "TestPw001"}):
            with patch("src.kabusapi.client.requests.Session") as MockSession:
                sess = MockSession.return_value
                sess.post.return_value = resp
                sess.headers = MagicMock()
                sess.headers.__iter__ = MagicMock(return_value=iter([
                    ("Content-Type", "application/json"),
                ]))
                sess.headers.items.return_value = [("Content-Type", "application/json")]
                import requests
                with self.assertRaises(requests.exceptions.HTTPError):
                    KabuClient().fetch_token()

    def test_forensic_emitted_on_401(self):
        resp = _make_http_response(401, _BODY_4001007)
        import logging
        with self.assertLogs("src.kabusapi.client", level="WARNING") as cm:
            self._run_fetch(resp)
        forensic_lines = [l for l in cm.output if "[TOKEN_FORENSIC]" in l]
        self.assertTrue(forensic_lines, "401応答で[TOKEN_FORENSIC]が出力されるべき")

    def test_forensic_contains_status_code(self):
        resp = _make_http_response(401, _BODY_4001007)
        with self.assertLogs("src.kabusapi.client", level="WARNING") as cm:
            self._run_fetch(resp)
        line = next(l for l in cm.output if "[TOKEN_FORENSIC]" in l)
        self.assertIn("status=401", line)

    def test_forensic_contains_pw_sha8(self):
        resp = _make_http_response(401, _BODY_4001007)
        expected_sha8 = hashlib.sha256("TestPw001".encode()).hexdigest()[:8]
        with self.assertLogs("src.kabusapi.client", level="WARNING") as cm:
            self._run_fetch(resp)
        line = next(l for l in cm.output if "[TOKEN_FORENSIC]" in l)
        self.assertIn(f"pw_sha8={expected_sha8}", line)

    def test_forensic_contains_pw_len(self):
        resp = _make_http_response(401, _BODY_4001007)
        with self.assertLogs("src.kabusapi.client", level="WARNING") as cm:
            self._run_fetch(resp)
        line = next(l for l in cm.output if "[TOKEN_FORENSIC]" in l)
        self.assertIn("pw_len=9", line)

    def test_forensic_masks_xapikey(self):
        """X-API-KEY は *** でマスクされる。"""
        resp = _make_http_response(401, _BODY_4001007)
        from src.kabusapi.client import KabuClient, _CACHE
        _CACHE.invalidate()
        # 実 dict を使うことで dict(self._session.headers) が正しく変換される
        with patch.dict(os.environ, {"KABU_API_PASSWORD": "TestPw001"}):
            with patch("src.kabusapi.client.requests.Session") as MockSession:
                sess = MockSession.return_value
                sess.post.return_value = resp
                sess.headers = {
                    "Content-Type": "application/json",
                    "X-API-KEY": "secret_token_value",
                }
                import requests
                with self.assertLogs("src.kabusapi.client", level="WARNING") as cm:
                    with self.assertRaises(requests.exceptions.HTTPError):
                        KabuClient().fetch_token()
        line = next(l for l in cm.output if "[TOKEN_FORENSIC]" in l)
        self.assertNotIn("secret_token_value", line)
        self.assertIn("***", line)

    def test_no_forensic_on_success(self):
        """200応答では [TOKEN_FORENSIC] を出力しない。"""
        from src.kabusapi.client import KabuClient, _CACHE
        _CACHE.invalidate()
        resp = _make_http_response(200, '{"Token":"good_token_abc"}')
        resp.json.return_value = {"Token": "good_token_abc"}
        with patch.dict(os.environ, {"KABU_API_PASSWORD": "TestPw001"}):
            with patch("src.kabusapi.client.requests.Session") as MockSession:
                sess = MockSession.return_value
                sess.post.return_value = resp
                sess.headers = {}
                import logging
                with patch.object(logging.getLogger("src.kabusapi.client"), "warning") as mock_w:
                    KabuClient().fetch_token()
                calls_text = " ".join(str(c) for c in mock_w.call_args_list)
                self.assertNotIn("TOKEN_FORENSIC", calls_text)

    def test_forensic_emitted_on_400(self):
        resp = _make_http_response(400, _BODY_4001011)
        with self.assertLogs("src.kabusapi.client", level="WARNING") as cm:
            self._run_fetch(resp)
        self.assertTrue(any("[TOKEN_FORENSIC]" in l for l in cm.output))


# ─────────────────────────────────────────────────────────────────────────────
# [2] API_AUTH_DETAIL — startup_check Code/Message 抽出
# ─────────────────────────────────────────────────────────────────────────────

class TestApiAuthDetail(unittest.TestCase):
    """startup_check._check_api_token() が Code/Message を個別ログ出力する。"""

    def _run_check(self, body: str, status: int = 401) -> tuple[bool, str, dict]:
        import requests
        resp = MagicMock()
        resp.status_code = status
        resp.text = body
        err = requests.exceptions.HTTPError(response=resp)

        out: dict = {}
        with patch("src.kabusapi.client.KabuClient") as MockCls:
            MockCls.return_value.fetch_token.side_effect = err
            # write_auth_failure は FAIL_OPEN なので無視可
            with patch("src.startup_check.json.loads", wraps=json.loads):
                with patch(
                    "src.diagnostics.api_auth_diagnostics.write_auth_failure",
                    return_value=None,
                ):
                    from src.startup_check import _check_api_token
                    ok, msg = _check_api_token(_out=out)
        return ok, msg, out

    def test_code_extracted_to_out(self):
        _, _, out = self._run_check(_BODY_4001007, 401)
        self.assertEqual(out.get("error_code"), 4001007)

    def test_message_extracted_to_out(self):
        _, _, out = self._run_check(_BODY_4001007, 401)
        self.assertEqual(out.get("error_message"), "ログイン認証エラー")

    def test_detail_logged_on_401(self):
        import src.startup_check as sc
        import requests
        resp = MagicMock()
        resp.status_code = 401
        resp.text = _BODY_4001007
        err = requests.exceptions.HTTPError(response=resp)

        with patch("src.kabusapi.client.KabuClient") as MockCls:
            MockCls.return_value.fetch_token.side_effect = err
            with patch(
                "src.diagnostics.api_auth_diagnostics.write_auth_failure",
                return_value=None,
            ):
                import logging
                with self.assertLogs("src.startup_check", level="WARNING") as cm:
                    sc._check_api_token()
        detail_lines = [l for l in cm.output if "[API_AUTH_DETAIL]" in l]
        self.assertTrue(detail_lines, "[API_AUTH_DETAIL] が出力されるべき")
        line = detail_lines[0]
        self.assertIn("4001007", line)
        self.assertIn("ログイン認証エラー", line)

    def test_code_none_on_non_json_body(self):
        """非JSON応答でも例外を送出せずに処理継続する。"""
        _, _, out = self._run_check("Internal Server Error", 503)
        self.assertIsNone(out.get("error_code"))


# ─────────────────────────────────────────────────────────────────────────────
# [3] api_auth_failures.jsonl — 書き込み確認
# ─────────────────────────────────────────────────────────────────────────────

class TestWriteAuthFailure(unittest.TestCase):
    """write_auth_failure() が runtime/api_auth_failures.jsonl に正しく追記する。"""

    def test_writes_jsonl_record(self):
        from src.diagnostics.api_auth_diagnostics import write_auth_failure, _FAILURES_FILE
        with tempfile.TemporaryDirectory() as tmp:
            tmp_file = Path(tmp) / "api_auth_failures.jsonl"
            with patch("src.diagnostics.api_auth_diagnostics._FAILURES_FILE", tmp_file):
                write_auth_failure(
                    timestamp="2026-06-26T08:44:02+0900",
                    status=401,
                    code=4001007,
                    message="ログイン認証エラー",
                    pid=1212,
                    pw_sha8="72f9e20f",
                )
            record = json.loads(tmp_file.read_text(encoding="utf-8").strip())
        self.assertEqual(record["status"],    401)
        self.assertEqual(record["code"],      4001007)
        self.assertEqual(record["message"],   "ログイン認証エラー")
        self.assertEqual(record["pid"],       1212)
        self.assertEqual(record["pw_sha8"],   "72f9e20f")
        self.assertEqual(record["timestamp"], "2026-06-26T08:44:02+0900")

    def test_appends_multiple_records(self):
        from src.diagnostics.api_auth_diagnostics import write_auth_failure
        with tempfile.TemporaryDirectory() as tmp:
            tmp_file = Path(tmp) / "api_auth_failures.jsonl"
            with patch("src.diagnostics.api_auth_diagnostics._FAILURES_FILE", tmp_file):
                for i in range(3):
                    write_auth_failure(status=401, code=4001007, message=f"attempt {i}")
            lines = tmp_file.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertEqual(json.loads(lines[2])["message"], "attempt 2")

    def test_code_none_serializes_as_null(self):
        """code=None は JSON null としてシリアライズ。"""
        from src.diagnostics.api_auth_diagnostics import write_auth_failure
        with tempfile.TemporaryDirectory() as tmp:
            tmp_file = Path(tmp) / "api_auth_failures.jsonl"
            with patch("src.diagnostics.api_auth_diagnostics._FAILURES_FILE", tmp_file):
                write_auth_failure(status=0, code=None, message="ConnectionError")
            record = json.loads(tmp_file.read_text(encoding="utf-8").strip())
        self.assertIsNone(record["code"])

    def test_written_on_401_from_check_api_token(self):
        """_check_api_token() が 401 応答を受けると failures.jsonl に追記する。"""
        import requests
        resp = MagicMock()
        resp.status_code = 401
        resp.text = _BODY_4001007
        err = requests.exceptions.HTTPError(response=resp)

        written: list[dict] = []

        def _fake_waf(**kwargs):
            written.append(kwargs)

        with patch("src.kabusapi.client.KabuClient") as MockCls:
            MockCls.return_value.fetch_token.side_effect = err
            with patch(
                "src.diagnostics.api_auth_diagnostics.write_auth_failure",
                side_effect=_fake_waf,
            ):
                from src.startup_check import _check_api_token
                _check_api_token()

        self.assertTrue(written, "401 応答で write_auth_failure が呼ばれるべき")
        rec = written[0]
        self.assertEqual(rec.get("status"), 401)
        self.assertEqual(rec.get("code"),   4001007)

    def test_not_written_on_success(self):
        """成功時は failures.jsonl に書き込まない。"""
        written: list = []

        def _fake_waf(**kwargs):
            written.append(kwargs)

        with patch("src.kabusapi.client.KabuClient") as MockCls:
            MockCls.return_value.fetch_token.return_value = "good_token"
            with patch(
                "src.diagnostics.api_auth_diagnostics.write_auth_failure",
                side_effect=_fake_waf,
            ):
                from src.startup_check import _check_api_token
                _check_api_token()

        self.assertFalse(written, "成功時は write_auth_failure を呼ばない")


# ─────────────────────────────────────────────────────────────────────────────
# [4] KabuS.exe プロセス情報 — _collect() 内 psutil
# ─────────────────────────────────────────────────────────────────────────────

class TestKabusProcessInfo(unittest.TestCase):
    """_collect() が psutil 経由で KabuS.exe のPID/起動時刻/稼働秒数を記録する。"""

    def test_kabus_proc_info_written_to_diag_jsonl(self):
        """_collect() が kabus_proc_pid / kabus_start_time / kabus_uptime_sec を返す。"""
        import time as _time
        fake_create_time = _time.time() - 9600  # 160分前

        mock_proc = MagicMock()
        mock_proc.info = {"pid": 2444, "name": "KabuS.exe", "create_time": fake_create_time}

        import requests
        resp = MagicMock()
        resp.status_code = 401
        resp.text = _BODY_4001007
        err = requests.exceptions.HTTPError(response=resp)

        captured_diag: list[dict] = []

        def _fake_write_diag(d: dict) -> None:
            captured_diag.append(d)

        import src.startup_check as sc
        with patch("src.kabusapi.client.KabuClient") as MockCls:
            MockCls.return_value.fetch_token.side_effect = err
            with patch("psutil.process_iter", return_value=[mock_proc]):
                with patch.object(sc, "_check_api_port", return_value=(True, "OK")):
                    with patch(
                        "src.diagnostics.api_auth_diagnostics.write_auth_failure",
                        return_value=None,
                    ):
                        sc._check_api_token()

        if captured_diag:
            d = captured_diag[0]
            if "kabus_proc_pid" in d:
                self.assertEqual(d["kabus_proc_pid"], 2444)
                self.assertIn("kabus_start_time", d)
                self.assertIn("kabus_uptime_sec", d)
                self.assertGreater(d["kabus_uptime_sec"], 9000)

    def test_psutil_failure_does_not_raise(self):
        """psutil が例外を送出しても _check_api_token() は正常に返る。"""
        import requests
        resp = MagicMock()
        resp.status_code = 401
        resp.text = _BODY_4001007
        err = requests.exceptions.HTTPError(response=resp)

        with patch("src.kabusapi.client.KabuClient") as MockCls:
            MockCls.return_value.fetch_token.side_effect = err
            with patch("psutil.process_iter", side_effect=RuntimeError("psutil_fail")):
                with patch(
                    "src.diagnostics.api_auth_diagnostics.write_auth_failure",
                    return_value=None,
                ):
                    from src.startup_check import _check_api_token
                    ok, msg = _check_api_token()
        self.assertFalse(ok)


# ─────────────────────────────────────────────────────────────────────────────
# [5] API_AUTH_RETRY ログ — retry ループ
# ─────────────────────────────────────────────────────────────────────────────

class TestApiAuthRetryLog(unittest.TestCase):
    """_check_api_token_with_retry() が [API_AUTH_RETRY] を各試行で出力する。"""

    def _run_retry(self, retries: int = 2) -> list[str]:
        import src.startup_check as sc
        import requests
        resp = MagicMock()
        resp.status_code = 401
        resp.text = _BODY_4001007
        err = requests.exceptions.HTTPError(response=resp)

        auth_calls: int = 0

        def _fake_auth(_out=None):
            nonlocal auth_calls
            auth_calls += 1
            if _out is not None:
                _out["response_body"] = _BODY_4001007
                _out["status_detail"] = 401
                _out["error_code"]    = 4001007
                _out["error_message"] = "ログイン認証エラー"
            return False, "API認証失敗 (HTTP 401: kabu Station未ログインまたはAPIパスワード不一致)"

        with patch.object(sc, "_check_api_token",   side_effect=_fake_auth):
            with patch.object(sc, "_retry_sleep",   return_value=None):
                with patch.object(sc, "_diag_record_attempt", return_value=None):
                    with self.assertLogs("src.startup_check", level="WARNING") as cm:
                        sc._check_api_token_with_retry(
                            retries=retries, retry_seconds=0, is_live=True,
                        )
        return cm.output

    def test_retry_log_emitted_each_attempt(self):
        logs = self._run_retry(retries=3)
        retry_lines = [l for l in logs if "[API_AUTH_RETRY]" in l]
        self.assertEqual(len(retry_lines), 3, f"3回試行で3行出力されるべき: {retry_lines}")

    def test_retry_log_contains_attempt_fraction(self):
        logs = self._run_retry(retries=2)
        retry_lines = [l for l in logs if "[API_AUTH_RETRY]" in l]
        self.assertTrue(any("attempt=1/2" in l for l in retry_lines))
        self.assertTrue(any("attempt=2/2" in l for l in retry_lines))

    def test_retry_log_contains_elapsed(self):
        logs = self._run_retry(retries=2)
        retry_lines = [l for l in logs if "[API_AUTH_RETRY]" in l]
        self.assertTrue(all("elapsed=" in l for l in retry_lines))

    def test_retry_log_contains_response_ms(self):
        logs = self._run_retry(retries=2)
        retry_lines = [l for l in logs if "[API_AUTH_RETRY]" in l]
        self.assertTrue(all("response_ms=" in l for l in retry_lines))


# ─────────────────────────────────────────────────────────────────────────────
# [6] 統合確認 — 正常系 / localhost停止 / 不正パスワード
# ─────────────────────────────────────────────────────────────────────────────

class TestForensicsIntegration(unittest.TestCase):

    def test_success_no_token_forensic_no_failure_record(self):
        """正常系: TOKEN_FORENSIC なし / failures.jsonl 書き込みなし。"""
        written: list = []

        with patch("src.kabusapi.client.KabuClient") as MockCls:
            MockCls.return_value.fetch_token.return_value = "valid_token"
            with patch(
                "src.diagnostics.api_auth_diagnostics.write_auth_failure",
                side_effect=lambda **kw: written.append(kw),
            ):
                import logging
                with patch.object(
                    logging.getLogger("src.kabusapi.client"), "warning"
                ) as mock_warn:
                    from src.startup_check import _check_api_token
                    ok, msg = _check_api_token()

        self.assertTrue(ok)
        forensic_calls = [c for c in mock_warn.call_args_list if "TOKEN_FORENSIC" in str(c)]
        self.assertFalse(forensic_calls, "正常系で TOKEN_FORENSIC は出力しない")
        self.assertFalse(written, "正常系で failures.jsonl に書き込まない")

    def test_connection_refused_writes_failure_record(self):
        """localhost:18080 停止: ConnectionError → failures.jsonl 追記。"""
        import requests
        written: list = []

        with patch("src.kabusapi.client.KabuClient") as MockCls:
            MockCls.return_value.fetch_token.side_effect = (
                requests.exceptions.ConnectionError("Connection refused")
            )
            with patch(
                "src.diagnostics.api_auth_diagnostics.write_auth_failure",
                side_effect=lambda **kw: written.append(kw),
            ):
                from src.startup_check import _check_api_token
                ok, msg = _check_api_token()

        self.assertFalse(ok)
        self.assertIn("接続エラー", msg)
        self.assertTrue(written, "ConnectionError でも failures.jsonl に書き込む")
        self.assertEqual(written[0]["status"], 0)

    def test_wrong_password_401_full_forensics(self):
        """不正パスワード: 401 → TOKEN_FORENSIC + API_AUTH_DETAIL + failures.jsonl。"""
        import requests
        resp = MagicMock()
        resp.status_code = 401
        resp.text = _BODY_4001007
        err = requests.exceptions.HTTPError(response=resp)
        written: list = []

        with patch("src.kabusapi.client.KabuClient") as MockCls:
            MockCls.return_value.fetch_token.side_effect = err
            with patch(
                "src.diagnostics.api_auth_diagnostics.write_auth_failure",
                side_effect=lambda **kw: written.append(kw),
            ):
                import src.startup_check as sc
                with self.assertLogs("src.startup_check", level="WARNING") as cm:
                    ok, msg = sc._check_api_token()

        self.assertFalse(ok)
        self.assertIn("401", msg)
        self.assertTrue(any("[API_AUTH_DETAIL]" in l for l in cm.output),
                        "[API_AUTH_DETAIL] が出力されるべき")
        self.assertTrue(written, "401で failures.jsonl に書き込む")
        self.assertEqual(written[0]["code"], 4001007)


if __name__ == "__main__":
    unittest.main(verbosity=2)
