"""
src/watchdog_runner.py
タスクスケジューラ用ラッパー。

- スタートアップチェック
- run_live_signal.py をサブプロセスで実行（LIVE_ALLOWED_SCRIPTSを通過させる）
- クラッシュ検知・ログ保存・通知送信
- ログ: logs/scheduler/YYYYMMDD_HHMMSS_{dry|live}.log

使い方 (タスクスケジューラから):
  8:43 ドライラン: python src\\watchdog_runner.py --dry
  8:44 ライブ実行: python src\\watchdog_runner.py --live --yes
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTHONUTF8", "1")  # propagate UTF-8 mode to child processes

import argparse
import json
import locale
import logging
import signal as _signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path

if os.name == "nt":
    locale.getpreferredencoding = lambda do_setlocale=True: "utf-8"

JST        = timezone(timedelta(hours=9))
_HERE      = Path(__file__).resolve().parent            # C:\ai-trading\src
_ROOT      = _HERE.parent                               # C:\ai-trading
_LOG_DIR   = _ROOT / "logs" / "scheduler"
_MAX_BODY  = 3000   # 通知メール本文の最大文字数


def _setup_logging(log_file: Path) -> None:
    """stdout と log_file の両方にログを出力するよう設定する。"""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def _run_script(extra_args: list[str], timeout: int = 1800) -> subprocess.CompletedProcess:
    """
    run_live_signal.py を独立サブプロセスで実行する。

    サブプロセスとして呼ぶ理由:
      paths.py の LIVE_ALLOWED_SCRIPTS チェックが sys.argv[0] ベースで動作するため、
      直接 import して呼ぶと "watchdog_runner.py" と判定されブロックされる。

    Popen + communicate() を明示使用する理由 (subprocess.run() から変更):
      subprocess.run() は communicate() を内部で呼ぶが proc オブジェクトを
      外部に公開しない。タイムアウト診断 ([PROC_STATE] / [COMMUNICATE_BEGIN/END])
      を行うために Popen を直接使用する。
    """
    target = _HERE / "run_live_signal.py"
    cmd = [sys.executable, str(target)] + extra_args
    logging.info("実行コマンド: %s", " ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(_ROOT),
        env=dict(os.environ, PYTHONUTF8="1"),
    )
    logging.warning(
        "[PROC_STATE] poll=%s returncode=%s pid=%s",
        proc.poll(), proc.returncode, proc.pid,
    )

    try:
        logging.warning("[COMMUNICATE_BEGIN] pid=%s timeout=%s", proc.pid, timeout)
        stdout_b, stderr_b = proc.communicate(timeout=timeout)
        logging.warning("[COMMUNICATE_END] pid=%s returncode=%s", proc.pid, proc.returncode)
    except subprocess.TimeoutExpired as _te:
        logging.warning(
            "[PROC_STATE] poll=%s returncode=%s pid=%s",
            proc.poll(),
            proc.returncode,
            proc.pid,
        )
        proc.kill()
        logging.warning("[WAIT_BEGIN] pid=%s", proc.pid)
        try:
            stdout_b, stderr_b = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            stdout_b, stderr_b = b"", b""
        logging.warning("[WAIT_END] pid=%s returncode=%s", proc.pid, proc.returncode)
        # pid を例外に付与して呼び出し元の child_pid 抽出を補助する
        _te._watchdog_pid = proc.pid  # type: ignore[attr-defined]
        raise

    stdout = stdout_b.decode("utf-8", errors="replace") if stdout_b else ""
    stderr = stderr_b.decode("utf-8", errors="replace") if stderr_b else ""
    return subprocess.CompletedProcess(
        args=cmd,
        returncode=proc.returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _truncate(text: str, max_chars: int = _MAX_BODY) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n...(省略)...\n" + text[-half:]


def _build_body(result: subprocess.CompletedProcess, check_result: dict, mode: str) -> str:
    ts = datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S JST")
    # Guard: subprocess.CompletedProcess.stdout/stderr are None when capture_output
    # was not used, or when the pipe was never written to. .strip() on None crashes.
    stdout_text = result.stdout or ""
    stderr_text = result.stderr or ""
    lines = [
        f"実行モード  : {mode.upper()}",
        f"実行時刻    : {ts}",
        f"終了コード  : {result.returncode}",
        f"ヘルスチェック: {check_result['summary']}",
        "",
    ]
    if check_result["warnings"]:
        lines.append("=== 警告 ===")
        lines.extend(f"  {w}" for w in check_result["warnings"])
        lines.append("")
    if stdout_text.strip():
        lines.append("=== 標準出力 ===")
        lines.append(_truncate(stdout_text.strip()))
        lines.append("")
    if stderr_text.strip():
        lines.append("=== 標準エラー ===")
        lines.append(_truncate(stderr_text.strip()))
    return "\n".join(lines)


def _release_stale_lock(lock_path: Path, pid: int | None) -> None:
    """タイムアウト後に陳腐化した order_lock.json を削除する。"""
    try:
        if not lock_path.exists():
            return
        data = json.loads(lock_path.read_text(encoding="utf-8"))
        # プロセスロック形式（"pid"キー）かつ PID が一致する場合のみ削除
        if "pid" in data and (pid is None or data.get("pid") == pid):
            lock_path.unlink(missing_ok=True)
            logging.warning(
                "タイムアウトクリーンアップ: 陳腐化ロック削除 %s (PID=%s)", lock_path, pid
            )
    except Exception as e:
        logging.warning("ロック削除失敗 (非致命的): %s", e)


def main() -> int:
    parser = argparse.ArgumentParser(description="CHIBA Trading Watchdog Runner")
    parser.add_argument("--dry",  action="store_true", help="ドライラン（発注なし）")
    parser.add_argument("--live", action="store_true", help="ライブ発注")
    parser.add_argument("--yes",  action="store_true", help="確認プロンプトをスキップ")
    args = parser.parse_args()

    # モード決定
    is_live = args.live and not args.dry
    mode    = "live" if is_live else "dry"

    # ログファイル設定
    ts_str   = datetime.now(JST).strftime("%Y%m%d_%H%M%S")
    log_file = _LOG_DIR / f"{ts_str}_{mode}.log"
    _setup_logging(log_file)

    logging.info("=" * 60)
    logging.info("CHIBA Watchdog Runner 起動 [%s]", mode.upper())
    logging.info("ログ: %s", log_file)
    logging.info(
        "[ENCODING] stdout=%s stderr=%s preferred=%s PYTHONUTF8=%s",
        getattr(sys.stdout, "encoding", "?"),
        getattr(sys.stderr, "encoding", "?"),
        locale.getpreferredencoding(False),
        os.environ.get("PYTHONUTF8", "0"),
    )

    # ── [1] スタートアップチェック ────────────────────────────────────────────
    try:
        sys.path.insert(0, str(_HERE))
        from startup_check import run_startup_check
        check_result = run_startup_check(is_live=is_live)
    except Exception:
        logging.error("スタートアップチェック例外:\n%s", traceback.format_exc())
        check_result = {"ok": False, "issues": ["startup_check 実行エラー"],
                        "warnings": [], "summary": "FAIL", "state": {},
                        "stale_data_detected": False, "snapshot_date": None, "expected_date": None}

    logging.info("スタートアップチェック: %s", check_result["summary"])
    for w in check_result["warnings"]:
        logging.warning(w)
    for e in check_result["issues"]:
        logging.error(e)

    # stale_data metrics 構造化ログ
    if check_result.get("stale_data_detected"):
        logging.error(
            "[STALE_DATA] stale_data_detected=True snapshot_date=%s expected_date=%s mode=%s",
            check_result.get("snapshot_date"), check_result.get("expected_date"), mode.upper(),
        )
    else:
        logging.info(
            "[STALE_DATA] stale_data_detected=False snapshot_date=%s expected_date=%s",
            check_result.get("snapshot_date"), check_result.get("expected_date"),
        )

    # api_auth metrics 構造化ログ
    _auth_ok  = check_result.get("api_auth_ok",  True)
    _auth_msg = check_result.get("api_auth_msg", "unknown")
    if _auth_ok:
        logging.info("[API_AUTH] ok=True msg=%s", _auth_msg)
    else:
        logging.error("[API_AUTH] ok=False msg=%s mode=%s", _auth_msg, mode.upper())

    if not check_result["ok"]:
        # Broker-as-Sole-SSOT (2026-07-18): LIVE→DRY への黙示的な降格は廃止。
        # 旧実装はAPI認証失敗のみ（ポート到達OK・他に致命的問題なし）の場合、
        # LIVEをDRYへ降格して実行を継続していたが、これは「brokerが本来疎通
        # すべきなのに疎通していない」異常を隠蔽する経路になり得た。
        # LIVE/DRY問わず、startup_checkが失敗した場合は即座に実行をブロックする
        # （fail-closed統一）。
        logging.error("スタートアップチェック失敗 → 実行をブロックします (mode=%s)", mode.upper())
        try:
            from notifier import notify_error, wait_pending
            notify_error(
                f"スタートアップチェック失敗のため{mode.upper()}実行を中止しました。\n\n"
                "問題:\n" + "\n".join(f"  - {e}" for e in check_result["issues"]),
                subject_suffix="スタートアップチェック失敗",
            )
            wait_pending(timeout=12)
        except Exception:
            pass
        return 1

    # ── [2] メインスクリプト実行 ─────────────────────────────────────────────
    script_args: list[str] = []
    if is_live:
        script_args += ["--live"]
    if args.yes:
        script_args += ["--yes"]

    _t_start = time.monotonic()
    _child_proc: subprocess.Popen | None = None  # populated inside _run_script via exception
    try:
        result = _run_script(script_args, timeout=1800)
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - _t_start
        child_pid: int | None = (
            getattr(exc, "_watchdog_pid", None)
            or getattr(exc, "pid", None)
            or (getattr(exc, "process", None) and exc.process.pid)  # type: ignore[union-attr]
        )

        # ── 構造化タイムアウト診断 ───────────────────────────────────────────
        logging.error(
            "タイムアウト: elapsed=%.1fs pid=%s phase=subprocess",
            elapsed, child_pid,
        )

        # ── 子プロセスを安全に強制終了 ─────────────────────────────────────
        try:
            if child_pid:
                import ctypes
                # Windows: TerminateProcess 相当。SIGKILL はない。
                subprocess.run(
                    ["taskkill", "/F", "/PID", str(child_pid)],
                    capture_output=True, timeout=5,
                )
                logging.warning("タイムアウトクリーンアップ: taskkill PID=%s", child_pid)
        except Exception as kill_err:
            logging.warning("子プロセス強制終了失敗 (非致命的): %s", kill_err)

        # ── 陳腐化ロックを解放 ──────────────────────────────────────────────
        try:
            from paths import EXECUTION_LOCK_FILE
            _release_stale_lock(EXECUTION_LOCK_FILE, child_pid)
        except Exception as lock_err:
            logging.warning("ロック解放スキップ: %s", lock_err)

        try:
            from notifier import notify_error, wait_pending
            notify_error(
                f"run_live_signal.py が1800秒でタイムアウトしました\n"
                f"elapsed={elapsed:.1f}s pid={child_pid}",
                "タイムアウト",
            )
            wait_pending(timeout=12)
        except Exception:
            pass
        return 1
    except Exception:
        logging.error("サブプロセス起動エラー:\n%s", traceback.format_exc())
        return 1

    # ── [3] 結果ログ ────────────────────────────────────────────────────────
    logging.info("終了コード: %d", result.returncode)
    if result.stdout:
        logging.info("STDOUT:\n%s", result.stdout[-4000:])
    if result.stderr:
        logging.warning("STDERR:\n%s", result.stderr[-2000:])

    # ── [3.5] 認証サマリ ─────────────────────────────────────────────────────
    try:
        from src.diagnostics.api_auth_diagnostics import log_auth_summary as _log_auth_summary
        _log_auth_summary(logging)
    except Exception:
        pass

    # ── [4] 通知 ────────────────────────────────────────────────────────────
    try:
        from notifier import notify_success, notify_warning, notify_error, notify_dry_run, wait_pending

        body = _build_body(result, check_result, mode)

        if mode == "dry":
            notify_dry_run(body)
        elif result.returncode == 0:
            if check_result["warnings"]:
                notify_warning(body, subject_suffix="発注完了（警告あり）")
            else:
                notify_success(body)
        else:
            notify_error(body, subject_suffix=f"exit={result.returncode}")

        wait_pending(timeout=20)
    except Exception:
        logging.warning("通知送信エラー (非致命的):\n%s", traceback.format_exc())

    return result.returncode


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
