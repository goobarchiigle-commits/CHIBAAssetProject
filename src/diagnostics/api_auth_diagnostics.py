"""
src/diagnostics/api_auth_diagnostics.py
API認証試行の観測性レイヤー。認証判定ロジックは一切変更しない。
FAIL_OPEN: 書き込みエラーはすべて無視。
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Union

_logger = logging.getLogger(__name__)
JST = timezone(timedelta(hours=9))

_HERE = Path(__file__).resolve().parent
# _HERE = src/diagnostics  → parents[1] = src  → parents[2] = C:\ai-trading (root)
_BASE = Path(os.environ.get("AI_TRADING_HOME", str(_HERE.parents[1])))

_DIAG_DIR      = _BASE / "runtime" / "diagnostics"
_HISTORY_FILE  = _DIAG_DIR / "api_auth_history.jsonl"
_SUMMARY_FILE  = _DIAG_DIR / "api_auth_summary.json"
_FAILURES_FILE = _BASE / "runtime" / "api_auth_failures.jsonl"

_MAX_HISTORY_BYTES = 5 * 1024 * 1024  # 5 MB


def _sha256_8(val: str) -> str:
    return hashlib.sha256(val.encode("utf-8")).hexdigest()[:8] if val else ""


# ─────────────────────────────────────────────────────────────────────────────
# 優先度A: 環境変数ソース特定
# ─────────────────────────────────────────────────────────────────────────────

def _detect_env_source(var_name: str = "KABU_API_PASSWORD") -> str:
    """
    var_name の値がどのソースから来たかを特定する。

    load_dotenv(override=False) の動作:
        プロセス環境変数(Windows env / Task Scheduler継承) > .env ファイル

    Returns:
        MACHINE_ENV  - HKLM レジストリの値と一致
        USER_ENV     - HKCU レジストリの値と一致
        DOTENV       - src/.env ファイルの値と一致
        PROCESS_ENV  - 上記どこにも一致しない（Task Scheduler独自設定等）
        NOT_SET      - 未設定
    """
    current_val = os.environ.get(var_name, "")
    if not current_val:
        return "NOT_SET"

    # ── HKLM (System-wide / Machine) ────────────────────────────────────────
    try:
        import winreg
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
        ) as key:
            try:
                machine_val, _ = winreg.QueryValueEx(key, var_name)
                if machine_val.strip() == current_val:
                    return "MACHINE_ENV"
            except FileNotFoundError:
                pass
    except Exception:
        pass

    # ── HKCU (User-level) ────────────────────────────────────────────────────
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as key:
            try:
                user_val, _ = winreg.QueryValueEx(key, var_name)
                if user_val.strip() == current_val:
                    return "USER_ENV"
            except FileNotFoundError:
                pass
    except Exception:
        pass

    # ── src/.env ファイル ────────────────────────────────────────────────────
    try:
        env_path = _BASE / "src" / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if line.startswith(f"{var_name}=") and not line.startswith("#"):
                    file_val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if file_val == current_val:
                        return "DOTENV"
    except Exception:
        pass

    # プロセス環境変数として存在するがレジストリ / .env と一致しない
    # → Task Scheduler の追加設定 or 親スクリプトで export された可能性
    return "PROCESS_ENV"


def _get_parent_pid() -> int:
    """Return parent process PID (0 on failure)."""
    try:
        return os.getppid()
    except AttributeError:
        return 0


def _get_task_name() -> str:
    """
    Task Scheduler タスク名を取得（環境変数 TASK_NAME があれば使用、なければ空文字）。
    Task Scheduler は既定では TASK_NAME を設定しない。
    設定したい場合はタスクの「開始オプション」で追加環境変数を設定すること。
    """
    return os.environ.get("TASK_NAME", "")


# ─────────────────────────────────────────────────────────────────────────────
# Task 2: JSONL 永続化
# ─────────────────────────────────────────────────────────────────────────────

def record_auth_attempt(
    *,
    status: Union[int, str],
    response_ms: float,
    retry_no: int,
    retry_max: int,
    api_port_ok: bool,
    api_auth_ok: bool,
    is_live: bool,
    mode: str,
    pw: str = "",
    startup_elapsed_sec: float = 0.0,
    base_url: str = "",
    ts: str = "",
    # 優先度A: 環境変数ソース
    env_source: str = "",
    parent_pid: int = 0,
    task_name: str = "",
    # 優先度B: 401レスポンスボディ
    response_body: str = "",
) -> None:
    """1認証試行を api_auth_history.jsonl に追記してサマリを更新（Fail-Open）。"""
    try:
        _DIAG_DIR.mkdir(parents=True, exist_ok=True)
        _ts = ts or datetime.now(JST).isoformat()
        _env_src = env_source or _detect_env_source()
        _ppid    = parent_pid or _get_parent_pid()
        _tname   = task_name  or _get_task_name()

        record: dict = {
            "ts":                  _ts,
            "status":              status,
            "response_ms":         round(response_ms, 1),
            "retry_no":            retry_no,
            "retry_max":           retry_max,
            "api_port_ok":         api_port_ok,
            "api_auth_ok":         api_auth_ok,
            "is_live":             is_live,
            "mode":                mode,
            "pw_sha256_8":         _sha256_8(pw),
            "pw_len":              len(pw),
            "pid":                 os.getpid(),
            "parent_pid":          _ppid,
            "user":                os.environ.get("USERNAME", "unknown"),
            "hostname":            socket.gethostname(),
            "startup_elapsed_sec": round(startup_elapsed_sec, 3),
            "base_url":            base_url,
            # 優先度A
            "env_source":          _env_src,
            "task_name":           _tname,
        }
        # 優先度B: response_body は 401 時のみ記録（空文字は省略）
        if response_body:
            record["response_body"] = response_body[:1000]

        _append_with_rotation(_HISTORY_FILE, record)
        _update_summary_file()
    except Exception as e:
        _logger.debug("[API_AUTH_DIAG] record_auth_attempt fail-open: %s", e)


def _append_with_rotation(path: Path, record: dict) -> None:
    """JSONL 追記。_MAX_HISTORY_BYTES 超過時は .1 にローテート。"""
    line = json.dumps(record, ensure_ascii=False) + "\n"
    if path.exists() and path.stat().st_size > _MAX_HISTORY_BYTES:
        rotated = path.with_suffix("").with_suffix(".jsonl.1")
        try:
            if rotated.exists():
                rotated.unlink()
            path.rename(rotated)
        except Exception:
            pass
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


# ─────────────────────────────────────────────────────────────────────────────
# Task 3: サマリ集計
# ─────────────────────────────────────────────────────────────────────────────

def _load_history() -> list[dict]:
    """JSONL 履歴を全読み込み（本体 + ローテート済み）。Fail-Open → []。"""
    records: list[dict] = []
    for p in [_HISTORY_FILE, _HISTORY_FILE.with_suffix("").with_suffix(".jsonl.1")]:
        if p.exists():
            try:
                for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            except Exception:
                pass
    return records


def _compute_summary(records: list[dict]) -> dict:
    """履歴レコード一覧からサマリ指標を計算。"""
    now    = datetime.now(JST)
    cutoff = (now - timedelta(days=7)).isoformat()

    sorted_recs = sorted(records, key=lambda r: r.get("ts", ""))

    last_success_ts = ""
    last_failure_ts = ""
    success_ms: list[float] = []
    failure_ms: list[float] = []

    for r in sorted_recs:
        ts_str = r.get("ts", "")
        ok     = r.get("api_auth_ok", False)
        ms     = float(r.get("response_ms", 0))
        in_7d  = ts_str >= cutoff
        if ok:
            last_success_ts = ts_str
            if in_7d:
                success_ms.append(ms)
        else:
            last_failure_ts = ts_str
            if in_7d:
                failure_ms.append(ms)

    # 末尾から連続失敗数をカウント
    consec = 0
    for r in reversed(sorted_recs):
        if r.get("api_auth_ok", False):
            break
        consec += 1

    return {
        "last_success_ts":           last_success_ts,
        "last_failure_ts":           last_failure_ts,
        "success_count_7d":          len(success_ms),
        "failure_count_7d":          len(failure_ms),
        "consecutive_failure_count": consec,
        "avg_response_ms_success":   round(sum(success_ms) / len(success_ms), 1) if success_ms else 0.0,
        "avg_response_ms_failure":   round(sum(failure_ms) / len(failure_ms), 1) if failure_ms else 0.0,
        "last_updated":              now.isoformat(),
    }


def _update_summary_file() -> None:
    """履歴から再計算してサマリ JSON をアトミック更新。"""
    records = _load_history()
    summary = _compute_summary(records)
    tmp = _SUMMARY_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(_SUMMARY_FILE)


def write_auth_failure(
    *,
    timestamp: str = "",
    status: int = 0,
    code: "int | None" = None,
    message: str = "",
    pid: int = 0,
    pw_sha8: str = "",
) -> None:
    """
    FAIL_OPEN。認証失敗1件を runtime/api_auth_failures.jsonl に追記。

    フォーマット:
      {"timestamp":"...", "status":401, "code":4001007, "message":"...", "pid":1234, "pw_sha8":"72f9e20f"}
    """
    try:
        record: dict = {
            "timestamp": timestamp or datetime.now(JST).isoformat(),
            "status":    status,
            "code":      code,
            "message":   message,
            "pid":       pid or os.getpid(),
            "pw_sha8":   pw_sha8,
        }
        _FAILURES_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_FAILURES_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        _logger.debug("[API_AUTH_FAILURES] write fail-open: %s", e)


def get_summary() -> dict:
    """api_auth_summary.json を読んで返す。不在・破損時は {}。"""
    if not _SUMMARY_FILE.exists():
        return {}
    try:
        return json.loads(_SUMMARY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Task 1: 構造化ログ出力
# ─────────────────────────────────────────────────────────────────────────────

def log_auth_diag(
    lgr: logging.Logger,
    *,
    retry_no: int,
    retry_max: int,
    status: Union[int, str],
    response_ms: float,
    api_port_ok: bool,
    api_auth_ok: bool,
    is_live: bool,
    mode: str,
    pw_len: int,
    pw_sha256_8: str,
    startup_elapsed_sec: float,
    pid: int,
    user: str,
    hostname: str,
    base_url: str,
    # 優先度A
    env_source: str = "",
    parent_pid: int = 0,
    task_name: str = "",
    # 優先度B
    response_body: str = "",
) -> None:
    """[API_AUTH_DIAG] 構造化ログを WARNING レベルで出力。パスワード平文は出力禁止。"""
    lgr.warning(
        "[API_AUTH_DIAG]"
        " ts=%s pid=%d user=%s host=%s"
        " startup_elapsed=%.1fs retry=%d/%d"
        " status=%s resp_ms=%.0f"
        " api_port_ok=%s api_auth_ok=%s"
        " is_live=%s mode=%s"
        " pw_len=%d pw_sha256_8=%s"
        " env_source=%s parent_pid=%d task_name=%s"
        " base_url=%s",
        datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z"),
        pid, user, hostname,
        startup_elapsed_sec,
        retry_no, retry_max,
        status, response_ms,
        api_port_ok, api_auth_ok,
        is_live, mode,
        pw_len, pw_sha256_8,
        env_source, parent_pid, task_name or "(none)",
        base_url,
    )
    if response_body:
        lgr.warning(
            "[API_AUTH_DIAG] response_body=%s",
            response_body[:500],
        )


# ─────────────────────────────────────────────────────────────────────────────
# Task 4: 日次サマリ出力
# ─────────────────────────────────────────────────────────────────────────────

def log_auth_summary(lgr: logging.Logger) -> None:
    """[API_AUTH_SUMMARY] 構造化ログを INFO レベルで出力。"""
    s = get_summary()
    if not s:
        lgr.info("[API_AUTH_SUMMARY] no data")
        return

    def _fmt(ts: str) -> str:
        if not ts:
            return "N/A"
        try:
            return ts[:19].replace("T", " ")
        except Exception:
            return "N/A"

    lgr.info(
        "[API_AUTH_SUMMARY]"
        " 7d_success=%d 7d_failure=%d"
        " consecutive_failures=%d"
        " last_success=%s last_failure=%s"
        " avg_success_ms=%.0f avg_failure_ms=%.0f",
        s.get("success_count_7d", 0),
        s.get("failure_count_7d", 0),
        s.get("consecutive_failure_count", 0),
        _fmt(s.get("last_success_ts", "")),
        _fmt(s.get("last_failure_ts", "")),
        s.get("avg_response_ms_success", 0.0),
        s.get("avg_response_ms_failure", 0.0),
    )
