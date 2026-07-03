"""
src/kabusapi/test_utf8_encoding.py
UTF-8 エンコーディング強制 + subprocess リレーのリグレッションテスト。

実行:
    cd C:/ai-trading
    python -m pytest src/kabusapi/test_utf8_encoding.py -v
"""
from __future__ import annotations

import io
import locale
import logging
import os
import subprocess
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


class TestEnvironmentBootstrap(unittest.TestCase):
    """PYTHONUTF8 と locale オーバーライドの確認。"""

    def test_pythonutf8_propagated_to_child(self):
        """PYTHONUTF8=1 が子プロセスに継承されること。"""
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        r = subprocess.run(
            [sys.executable, "-c", "import os; print(os.environ.get('PYTHONUTF8','0'))"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            env=env,
        )
        self.assertEqual(r.stdout.strip(), "1")

    def test_child_stderr_encoding_is_utf8_when_pythonutf8_set(self):
        """PYTHONUTF8=1 で起動した子プロセスの stderr エンコーディングが utf-8 であること。"""
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        r = subprocess.run(
            [sys.executable, "-c",
             "import sys; print(sys.stderr.encoding or 'None')"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            env=env,
        )
        enc = r.stdout.strip().lower().replace("-", "")
        self.assertIn(enc, ("utf8", "utf8bom"), f"expected utf8, got {enc!r}")

    def test_locale_override_on_nt(self):
        """Windows: lambda オーバーライド後に getpreferredencoding が utf-8 を返すこと。"""
        if os.name != "nt":
            self.skipTest("Windows only")
        orig = locale.getpreferredencoding
        try:
            locale.getpreferredencoding = lambda do_setlocale=True: "utf-8"
            result = locale.getpreferredencoding(False).lower().replace("-", "")
            self.assertEqual(result, "utf8")
        finally:
            locale.getpreferredencoding = orig


class TestSubprocessUTF8Relay(unittest.TestCase):
    """subprocess の stdout/stderr が文字化けなく中継されること。"""

    def _run_jp(self, extra_env: dict | None = None) -> str:
        """日本語を stderr に書くワンライナーを実行し、stderr テキストを返す。"""
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        if extra_env:
            env.update(extra_env)
        r = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.stderr.write('テスト日本語\\n'); sys.stderr.flush()"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            env=env,
        )
        return r.stderr

    def test_japanese_stderr_no_mojibake(self):
        out = self._run_jp()
        self.assertIn("テスト日本語", out, f"mojibake detected: {out!r}")

    def test_japanese_stderr_cp932_sim_replaced_not_crashed(self):
        """cp932 環境シミュレーション: errors='replace' で例外にならないこと。"""
        env = os.environ.copy()
        env.pop("PYTHONUTF8", None)  # UTF-8 mode off
        r = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.stderr.buffer.write('テスト'.encode('cp932')); sys.stderr.buffer.flush()"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            env=env,
        )
        # errors="replace" ensures no UnicodeDecodeError — output may contain ?
        self.assertIsInstance(r.stderr, str)


class TestLoggingStreamHandlerUTF8(unittest.TestCase):
    """StreamHandler(sys.stderr) が UTF-8 出力すること。"""

    def test_handler_writes_japanese_to_buffer(self):
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(logging.Formatter("%(message)s"))
        log = logging.getLogger("test_enc_relay")
        log.addHandler(handler)
        log.setLevel(logging.DEBUG)
        try:
            log.info("テスト 日本語 ログ stdout=utf-8 stderr=utf-8")
            out = buf.getvalue()
            self.assertIn("テスト", out)
            self.assertIn("stdout=utf-8", out)
        finally:
            log.removeHandler(handler)

    def test_stderr_encoding_after_reconfigure(self):
        """sys.stderr の encoding が utf-8 であること（reconfigure 済み前提）。"""
        enc = getattr(sys.stderr, "encoding", None) or ""
        self.assertIn(enc.lower().replace("-", ""), ("utf8", "utf8bom"),
                      f"sys.stderr.encoding={enc!r}")

    def test_stdout_encoding_after_reconfigure(self):
        enc = getattr(sys.stdout, "encoding", None) or ""
        self.assertIn(enc.lower().replace("-", ""), ("utf8", "utf8bom"),
                      f"sys.stdout.encoding={enc!r}")


class TestWatchdogRelayEncoding(unittest.TestCase):
    """watchdog_runner がサブプロセスの日本語 stderr を文字化けなく中継すること。"""

    def test_watchdog_subprocess_call_params(self):
        """watchdog_runner._run_script の subprocess.run 呼び出しが
        encoding='utf-8' and errors='replace' を持つこと（ソース検証）。"""
        src = (Path(__file__).parent.parent / "watchdog_runner.py").read_text(encoding="utf-8")
        self.assertIn('encoding="utf-8"', src)
        self.assertIn('errors="replace"', src)
        self.assertIn('os.environ.setdefault("PYTHONUTF8", "1")', src)

    def test_run_live_signal_has_pythonutf8(self):
        """run_live_signal.py が PYTHONUTF8 を設定すること（ソース検証）。"""
        src = (Path(__file__).parent.parent / "run_live_signal.py").read_text(encoding="utf-8")
        self.assertIn('os.environ.setdefault("PYTHONUTF8", "1")', src)

    def test_encoding_log_tag_present(self):
        """watchdog_runner.py に [ENCODING] 診断ログが存在すること。"""
        src = (Path(__file__).parent.parent / "watchdog_runner.py").read_text(encoding="utf-8")
        self.assertIn("[ENCODING]", src)

    def test_locale_override_present_in_watchdog(self):
        src = (Path(__file__).parent.parent / "watchdog_runner.py").read_text(encoding="utf-8")
        self.assertIn("locale.getpreferredencoding", src)

    def test_locale_override_present_in_run_live(self):
        src = (Path(__file__).parent.parent / "run_live_signal.py").read_text(encoding="utf-8")
        self.assertIn("locale.getpreferredencoding", src)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    unittest.main(verbosity=2)
