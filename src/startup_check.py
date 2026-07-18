"""
src/startup_check.py
毎朝実行前のヘルスチェック。副作用なし（読み取り専用）。

チェック項目:
  [1] kabuステーション API ポート到達確認 (localhost:18080)
  [2] portfolio_state.json 整合性（存在・JSON妥当性・必須キー・数値）
  [3] ドローダウン水準（-15%でBUY_STOP警告）
  [4] order_lock.json 陳腐化チェック（1時間超 → 前回クラッシュ疑い）
  [5] equity_peak vs available_cash 整合性
  [6] RSRスナップショット日付チェック（stale data FAIL-CLOSED）
      LIVE: snapshot < 前営業日 → issues 追加 → 発注停止
      DRY : snapshot < 前営業日 → warnings 追加 → 実行継続
      snapshot >= 前営業日（当日付含む）は常に OK
  [7] API 認証プリフライト（fetch_token FAIL-CLOSED）
      LIVE: 認証失敗 → issues 追加 → 発注停止
      DRY : 認証失敗 → warnings 追加 → 実行継続
      ポート到達不可の場合はスキップ（[1] が先に FAIL）

Returns: {"ok": bool, "issues": list[str], "warnings": list[str], "summary": str}
  ok=True  → 全チェック通過（warningsあっても発注継続可能）
  ok=False → 致命的問題あり（ライブ発注をブロックすべき）
"""
from __future__ import annotations

import json
import logging
import math
import os
import socket
import time
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional

_logger = logging.getLogger(__name__)


def _retry_sleep(seconds: float) -> None:
    """Sleep between retry attempts.

    No-op when running under pytest (PYTEST_CURRENT_TEST set) so that
    existing tests that do NOT patch this function remain fast.
    Production code always sleeps for the full duration.
    Patch `src.startup_check._retry_sleep` with a MagicMock() in tests
    that need to assert call counts / arguments.
    """
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return
    time.sleep(seconds)

from src.execution.dd_engine import compute_drawdown

JST = timezone(timedelta(hours=9))

_HERE      = Path(__file__).resolve().parent
_BASE_DIR  = Path(os.environ.get("AI_TRADING_HOME", str(_HERE.parent)))
_RUNTIME   = _BASE_DIR / "runtime"

_PORTFOLIO_FILE  = _RUNTIME / "portfolio_state.json"
_LOCK_FILE       = _RUNTIME / "order_lock.json"        # duplicate-order ledger (informational)
_EXEC_LOCK_FILE  = _RUNTIME / "execution.lock.json"    # execution mutex (staleness check)

_KABUS_HOST = os.environ.get("KABUS_HOST", "localhost")
_KABUS_PORT = int(os.environ.get("KABUS_PORT", "18080"))

_DD_WARN_THRESHOLD  = -0.15   # -15%: BUY_STOP 警告レベル（CLAUDE.md準拠）
_LOCK_STALE_SEC     = 3600    # 1時間超でロックファイル陳腐化とみなす

_REQUIRED_STATE_KEYS = frozenset({
    "cb_state", "equity_peak", "available_cash",
})
# last_equity はオプション（旧 state との後方互換）
_OPTIONAL_STATE_KEYS = frozenset({"last_equity", "safe_warn_count"})

_RSR_SNAPSHOT_DIR = _RUNTIME / "rsr"

# TSE 国民の祝日（土日を除く）2024–2026。SimpleTSECalendar と同期。
_TSE_HOLIDAYS: frozenset[str] = frozenset({
    "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-08",
    "2024-02-11", "2024-02-12", "2024-02-23",
    "2024-03-20", "2024-04-29",
    "2024-05-03", "2024-05-04", "2024-05-05", "2024-05-06",
    "2024-07-15", "2024-08-11", "2024-08-12",
    "2024-09-16", "2024-09-22", "2024-09-23",
    "2024-10-14", "2024-11-03", "2024-11-04", "2024-11-23",
    "2024-12-31",
    "2025-01-01", "2025-01-02", "2025-01-03", "2025-01-13",
    "2025-02-11", "2025-02-23", "2025-02-24",
    "2025-03-20", "2025-04-29",
    "2025-05-03", "2025-05-04", "2025-05-05", "2025-05-06",
    "2025-07-21", "2025-08-11",
    "2025-09-15", "2025-09-23", "2025-10-13",
    "2025-11-03", "2025-11-23", "2025-11-24",
    "2025-12-31",
    "2026-01-01", "2026-01-02", "2026-01-03", "2026-01-12",
    "2026-02-11", "2026-02-23",
    "2026-03-20", "2026-04-29",
    "2026-05-03", "2026-05-04", "2026-05-05", "2026-05-06",
    "2026-07-20", "2026-08-11",
    "2026-09-21", "2026-09-23", "2026-10-12",
    "2026-11-03", "2026-11-23",
    "2026-12-31",
})


def _is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d.strftime("%Y-%m-%d") not in _TSE_HOLIDAYS


def _prev_trading_day(today: date) -> date:
    """前営業日（土日・祝日を除く）を返す。"""
    d = today - timedelta(days=1)
    while not _is_trading_day(d):
        d -= timedelta(days=1)
    return d


# ── 内部チェック関数 ─────────────────────────────────────────────────────────

def _check_api_port() -> tuple[bool, str]:
    """[1] kabuステーション APIポート到達確認。"""
    try:
        with socket.create_connection((_KABUS_HOST, _KABUS_PORT), timeout=3):
            return True, f"API port {_KABUS_PORT} OK"
    except OSError as e:
        return False, f"API port {_KABUS_PORT} 到達不可: {e}"


def _check_api_token(_out: "dict | None" = None) -> tuple[bool, str]:
    """
    [7] API 認証プリフライト + 詳細診断ログ（一時追加・削除予定）

    fetch_token() を実行して kabu Station のブローカーログイン完了を確認する。
    ポート疎通成功後に呼ぶこと（_check_api_port が True の場合のみ）。

    401: kabu Station 起動中だがブローカーログイン未完了 or APIパスワード不一致。
    その他 HTTPError: サービス異常（503 等）。
    token 空文字: kabu Station が正常応答を返せない状態。

    診断ログ: logs/api_auth_diag.jsonl に成功・失敗問わず追記する。
    記録項目: status_code / resp_text / token_url / pid / ppid / win_user /
              cwd / sys_executable / pw_len / pw_sha256_8 /
              kabus_pid_listening / netstat_18080

    Args:
        _out: Optional mutable dict. Populated with observability metadata
              (response_body, status_detail) without affecting return values.

    Returns:
        (auth_ok, message)
        auth_ok=True  → token 取得成功
        auth_ok=False → 認証失敗（message に原因区分を含む）
    """
    import hashlib   as _hashlib
    import subprocess as _subprocess
    import sys        as _sys

    try:
        import requests as _req
    except ImportError:
        return False, "requests 未インストール — 認証チェック不可"

    # ── 診断情報収集 ──────────────────────────────────────────────────────────
    def _collect(http_status: object, resp_text: "str | None") -> dict:
        try:
            from dotenv import load_dotenv as _ld
            _ld(dotenv_path=_BASE_DIR / "src" / ".env", override=False)
        except ImportError:
            pass
        _pw   = os.getenv("KABU_API_PASSWORD", "")
        _port = os.getenv("KABU_API_PORT", "18080")

        pid = os.getpid()
        try:
            ppid: object = os.getppid()
        except AttributeError:
            ppid = "N/A"

        pw_hash8 = (
            _hashlib.sha256(_pw.encode("utf-8")).hexdigest()[:8] if _pw else "(empty)"
        )

        netstat_lines: list[str] = []
        kabus_pid = "unknown"
        try:
            _ns = _subprocess.run(
                ["netstat", "-ano"], capture_output=True, text=True, timeout=5,
            )
            for _ln in _ns.stdout.splitlines():
                if ":18080" in _ln:
                    netstat_lines.append(_ln.strip())
                    if "LISTENING" in _ln:
                        _parts = _ln.split()
                        if _parts:
                            kabus_pid = _parts[-1]
        except Exception as _ne:
            netstat_lines = [f"netstat_error:{_ne}"]

        # [4] KabuS.exe プロセス情報（401時のみ有意・FAIL_OPEN）
        kabus_proc: dict = {}
        try:
            import psutil as _psutil
            _now_ts = time.time()
            for _proc in _psutil.process_iter(["pid", "name", "create_time"]):
                try:
                    if "kabu" in (_proc.info["name"] or "").lower():
                        _ct = _proc.info["create_time"]
                        kabus_proc = {
                            "kabus_proc_pid":    _proc.info["pid"],
                            "kabus_start_time":  datetime.fromtimestamp(_ct, tz=JST).isoformat(),
                            "kabus_uptime_sec":  int(_now_ts - _ct),
                        }
                        break
                except (_psutil.NoSuchProcess, _psutil.AccessDenied):
                    pass
        except Exception as _pe:
            _logger.warning("[API_AUTH_DIAG] psutil KabuS info fail-open: %s", _pe)

        d = {
            "ts":                  datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S JST"),
            "http_status":         http_status,
            "resp_text":           resp_text,
            "token_url":           f"http://localhost:{_port}/kabusapi/token",
            "pid":                 pid,
            "ppid":                str(ppid),
            "win_user":            os.environ.get("USERNAME", "unknown"),
            "cwd":                 os.getcwd(),
            "sys_executable":      _sys.executable,
            "pw_len":              len(_pw),
            "pw_sha256_8":         pw_hash8,
            "kabus_pid_listening": kabus_pid,
            "netstat_18080":       netstat_lines,
        }
        if kabus_proc:
            d.update(kabus_proc)
        return d

    def _write_diag(d: dict) -> None:
        try:
            _p = _BASE_DIR / "logs" / "api_auth_diag.jsonl"
            _p.parent.mkdir(parents=True, exist_ok=True)
            with open(_p, "a", encoding="utf-8") as _f:
                _f.write(json.dumps(d, ensure_ascii=False) + "\n")
        except Exception:
            pass  # 診断ログ書き込み失敗は非致命的

    # ── 認証チェック（KabuClient 経由・既存テスト互換性を維持）─────────────
    try:
        from src.kabusapi.client import KabuClient
        token = KabuClient().fetch_token()
        if not token:
            _write_diag(_collect(200, '{"Token":""}'))
            return False, "API token empty"
        _write_diag(_collect(200, '{"Token":"[OK:masked]"}'))
        return True, "API token OK"
    except ValueError as e:
        _write_diag(_collect("ValueError", str(e)[:200]))
        return False, f"API認証不可 (パスワード未設定): {e}"
    except _req.exceptions.HTTPError as e:
        _status   = e.response.status_code if e.response is not None else "?"
        _rtext    = e.response.text[:1000] if e.response is not None else ""
        _write_diag(_collect(_status, _rtext))
        if _out is not None:
            _out["response_body"]  = _rtext
            _out["status_detail"]  = _status

        # [2] Code / Message を個別抽出
        _code: "int | None" = None
        _message: str = ""
        try:
            _parsed = json.loads(_rtext)
            _code    = _parsed.get("Code")
            _message = str(_parsed.get("Message", ""))
        except Exception:
            pass
        if _code is not None or _message:
            _logger.warning(
                "[API_AUTH_DETAIL] status=%s code=%s message=%s",
                _status, _code, _message,
            )
        if _out is not None:
            _out["error_code"]    = _code
            _out["error_message"] = _message

        # [3] runtime/api_auth_failures.jsonl 追記（FAIL_OPEN）
        try:
            from src.diagnostics.api_auth_diagnostics import write_auth_failure as _waf
            import hashlib as _hl
            _pw_for_hash = os.getenv("KABU_API_PASSWORD", "")
            _waf(
                status  = _status if isinstance(_status, int) else 0,
                code    = _code,
                message = _message,
                pw_sha8 = _hl.sha256(_pw_for_hash.encode()).hexdigest()[:8] if _pw_for_hash else "",
            )
        except Exception:
            pass

        if _status == 401:
            return False, (
                "API認証失敗 (HTTP 401: "
                "kabu Station未ログインまたはAPIパスワード不一致)"
            )
        return False, f"API認証失敗 (HTTP {_status})"
    except _req.exceptions.ConnectionError as e:
        _write_diag(_collect("ConnectionError", str(e)[:200]))
        # [3] 接続失敗も記録
        try:
            from src.diagnostics.api_auth_diagnostics import write_auth_failure as _waf
            _waf(status=0, code=None, message=f"ConnectionError: {str(e)[:200]}")
        except Exception:
            pass
        return False, f"API認証失敗 (接続エラー): {e}"
    except _req.exceptions.Timeout:
        _write_diag(_collect("Timeout", None))
        # [3] タイムアウトも記録
        try:
            from src.diagnostics.api_auth_diagnostics import write_auth_failure as _waf
            _waf(status=0, code=None, message="Timeout")
        except Exception:
            pass
        return False, "API認証失敗 (タイムアウト)"
    except Exception as e:
        _write_diag(_collect(f"{type(e).__name__}", str(e)[:200]))
        return False, f"API認証失敗 ({type(e).__name__}): {e}"


def _parse_status(ok: bool, msg: str) -> "int | str":
    """Return HTTP status code (or error category string) from auth result."""
    import re as _re
    if ok:
        return 200
    m = _re.search(r"HTTP (\d{3})", msg)
    if m:
        return int(m.group(1))
    if "接続エラー" in msg:
        return "ConnectionError"
    if "タイムアウト" in msg:
        return "Timeout"
    if "パスワード未設定" in msg:
        return "ValueError"
    return "unknown"


def _diag_record_attempt(
    *,
    ok: bool,
    msg: str,
    retry_no: int,
    retry_max: int,
    resp_ms: float,
    elapsed: float,
    is_live: bool,
    response_body: str = "",
) -> None:
    """Fail-Open observability hook — called once per auth attempt."""
    try:
        import hashlib as _hl
        from src.diagnostics.api_auth_diagnostics import (
            record_auth_attempt  as _record,
            log_auth_diag        as _log_diag,
            _detect_env_source,
            _get_parent_pid,
            _get_task_name,
        )
        _pw         = os.getenv("KABU_API_PASSWORD", "")
        _mode       = "live" if is_live else "dry"
        _status     = _parse_status(ok, msg)
        _env_source = _detect_env_source()
        _ppid       = _get_parent_pid()
        _tname      = _get_task_name()
        try:
            from src.kabusapi.client import BASE_URL as _base_url
        except Exception:
            _base_url = "http://localhost:18080/kabusapi"

        _record(
            status=_status,
            response_ms=resp_ms,
            retry_no=retry_no,
            retry_max=retry_max,
            api_port_ok=True,
            api_auth_ok=ok,
            is_live=is_live,
            mode=_mode,
            pw=_pw,
            startup_elapsed_sec=elapsed,
            base_url=_base_url,
            env_source=_env_source,
            parent_pid=_ppid,
            task_name=_tname,
            response_body=response_body,
        )
        if not ok:
            _log_diag(
                _logger,
                retry_no=retry_no,
                retry_max=retry_max,
                status=_status,
                response_ms=resp_ms,
                api_port_ok=True,
                api_auth_ok=False,
                is_live=is_live,
                mode=_mode,
                pw_len=len(_pw),
                pw_sha256_8=_hl.sha256(_pw.encode("utf-8")).hexdigest()[:8] if _pw else "",
                startup_elapsed_sec=elapsed,
                pid=os.getpid(),
                user=os.environ.get("USERNAME", "unknown"),
                hostname=__import__("socket").gethostname(),
                base_url=_base_url,
                env_source=_env_source,
                parent_pid=_ppid,
                task_name=_tname,
                response_body=response_body,
            )
    except Exception as _e:
        _logger.debug("[API_AUTH_DIAG] _diag_record_attempt fail-open: %s", _e)


def _check_api_token_with_retry(
    retries: int = 6,
    retry_seconds: int = 30,
    is_live: bool = False,
    start_ts: float = 0.0,
) -> tuple[bool, str]:
    """
    API認証プリフライト。

    HTTP 401 / ConnectionError / Timeout の場合のみ
    最大 retries 回まで再試行する。

    総待機時間: retry_seconds × (retries - 1) 秒（最大150秒）

    FAIL_CLOSEDは維持。
    テスト時は `src.startup_check._retry_sleep` をモックしてスリップを省略できる。
    """
    if start_ts == 0.0:
        start_ts = time.monotonic()
    last_msg = ""

    for attempt in range(retries):
        _t0   = time.monotonic()
        _tout: dict = {}
        ok, msg = _check_api_token(_out=_tout)
        _resp_ms = (time.monotonic() - _t0) * 1000
        _elapsed = time.monotonic() - start_ts
        _resp_body = _tout.get("response_body", "")

        # Observability hook (fail-open — never affects auth judgment)
        try:
            _diag_record_attempt(
                ok=ok, msg=msg,
                retry_no=attempt + 1, retry_max=retries,
                resp_ms=_resp_ms, elapsed=_elapsed,
                is_live=is_live,
                response_body=_resp_body,
            )
        except Exception:
            pass

        if ok:
            if attempt == 0:
                return True, msg
            _logger.info(
                "[API_AUTH] ok=True retry_success attempt=%d/%d",
                attempt + 1, retries,
            )
            return True, f"{msg} (retry_success attempt={attempt + 1}/{retries})"

        last_msg = msg
        _logger.warning(
            "[API_AUTH] attempt %d/%d failed: %s%s",
            attempt + 1, retries, msg,
            f" — retrying in {retry_seconds}s" if attempt < retries - 1 else "",
        )
        _logger.warning(
            "[API_AUTH_RETRY] attempt=%d/%d elapsed=%.0fs response_ms=%.0f",
            attempt + 1, retries, _elapsed, _resp_ms,
        )

        if attempt < retries - 1:
            _retry_sleep(retry_seconds)

    _logger.error(
        "[API_AUTH] ok=False retry_exhausted %d/%d: %s",
        retries, retries, last_msg,
    )
    return False, f"{last_msg} (retry_exhausted {retries}/{retries})"


def _check_portfolio_state() -> tuple[bool, list[str], list[str], dict[str, Any]]:
    """
    [2][3][5] portfolio_state.json 整合性チェック。

    Returns:
        (ok, issues, warnings, state_dict)
    """
    issues:   list[str] = []
    warnings: list[str] = []
    state:    dict[str, Any] = {}

    # ファイル存在確認
    if not _PORTFOLIO_FILE.exists():
        issues.append(f"portfolio_state.json が見つかりません: {_PORTFOLIO_FILE}")
        return False, issues, warnings, state

    # JSON パース
    try:
        raw = _PORTFOLIO_FILE.read_text(encoding="utf-8")
        state = json.loads(raw)
    except json.JSONDecodeError as e:
        issues.append(f"portfolio_state.json が壊れています (JSON parse error): {e}")
        return False, issues, warnings, state

    # 必須キー確認
    missing = _REQUIRED_STATE_KEYS - set(state.keys())
    if missing:
        issues.append(f"portfolio_state.json に必須キーが不足: {sorted(missing)}")
        return False, issues, warnings, state

    # 数値チェック
    equity_peak = state.get("equity_peak", 0)
    available_cash = state.get("available_cash", 0)

    try:
        ep = float(equity_peak)
        ac = float(available_cash)
    except (TypeError, ValueError) as e:
        issues.append(f"portfolio_state.json に非数値: {e}")
        return False, issues, warnings, state

    if not math.isfinite(ep) or ep <= 0:
        issues.append(f"equity_peak 不正値: {ep}")
        return False, issues, warnings, state

    if not math.isfinite(ac) or ac < 0:
        issues.append(f"available_cash 不正値: {ac}")
        return False, issues, warnings, state

    # [5a] SAFE_WARN 状態チェック
    cb_state_now = str(state.get("cb_state", "NORMAL"))
    if cb_state_now == "SAFE_WARN":
        swc = int(state.get("safe_warn_count", 0))
        warnings.append(
            f"⚠️ SAFE_WARN: equity_peak 異常検出済み "
            f"(確認={swc}回 / 昇格閾値=3回)。BUY は継続中。"
        )

    # [3][5b] DD水準チェック・PEAK_ANOMALY は run_startup_check() で
    # compute_live_equity(OHLCV) ベースの current_equity を使って実施する。
    # last_equity（前回 run 書き込み値）への依存を廃止。

    # [5c] equity_peak vs available_cash 整合性（逆転は異常）
    if ac > ep * 1.01:  # 1%の誤差は許容
        warnings.append(
            f"available_cash (¥{ac:,.0f}) > equity_peak (¥{ep:,.0f}) × 1.01 — 状態不整合の可能性"
        )

    return True, issues, warnings, state


def _check_lock_file() -> tuple[bool, str]:
    """
    [4] execution.lock.json チェック（実行 mutex の陳腐化検出）。

    execution.lock.json は実行 mutex 専用ファイル。
    "pid" + "heartbeat_ts" キーが揃っている場合のみ mutex 形式として扱う。
    それ以外（非 mutex アーティファクト）はスキップ（正常）。
    1時間超過の場合のみ警告（前回クラッシュ疑い）。

    order_lock.json は発注履歴台帳のため、このチェックの対象外。
    """
    if not _EXEC_LOCK_FILE.exists():
        return True, "execution.lock: なし（正常）"
    try:
        data = json.loads(_EXEC_LOCK_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        return True, f"execution.lock parse skip: {e}"

    # mutex 形式かどうかを判定
    if "pid" not in data or "heartbeat_ts" not in data:
        return True, "execution.lock: 非 mutex アーティファクト（スキップ）"

    # heartbeat_ts ベースで陳腐化チェック
    hb = data.get("heartbeat_ts")
    try:
        age_sec = time.time() - float(hb) if hb is not None else _LOCK_STALE_SEC + 1
    except (TypeError, ValueError):
        return True, "execution.lock: heartbeat_ts 値不正、スキップ"

    if age_sec > _LOCK_STALE_SEC:
        pid = data.get("pid", "?")
        return False, (
            f"execution.lock.json が陳腐化 ({age_sec/3600:.1f}h経過, PID={pid}) — "
            f"前回クラッシュの疑い。{_EXEC_LOCK_FILE} を確認後、次回起動で自動回収されます。"
        )
    return True, f"execution.lock: PID={data.get('pid','?')}, {age_sec:.0f}s前"


def _check_snapshot_date(
    today: Optional[date] = None,
    rsr_dir: Optional[Path] = None,
) -> tuple[bool, str, Optional[str], Optional[str]]:
    """
    [6] RSR スナップショット日付チェック（Stale Market Data）。

    runtime/rsr/ 内の最新 YYYY-MM-DD.json のステム日付を snapshot_date とし、
    expected_date = 前営業日 と日付型で比較する。

    snapshot_date >= expected_date → OK（当日付・非営業日付も許容）
    snapshot_date <  expected_date → stale（古すぎるデータ）

    Returns:
        (stale_detected, message, snapshot_date, expected_date)
        stale_detected=True → caller が is_live に応じて issue / warning に振り分ける。
        snapshot_date / expected_date は None if rsr dir missing or empty。
    """
    _today = today or datetime.now(JST).date()
    _rsr_dir = rsr_dir if rsr_dir is not None else _RSR_SNAPSHOT_DIR

    if not _rsr_dir.exists():
        return False, f"RSR snapshot dir なし ({_rsr_dir}) — チェックをスキップ", None, None

    snaps = sorted(_rsr_dir.glob("*.json"))
    if not snaps:
        return False, "RSR snapshot 未作成 — チェックをスキップ", None, None

    snapshot_date_str = snaps[-1].stem          # YYYY-MM-DD
    expected = _prev_trading_day(_today)
    expected_str = expected.strftime("%Y-%m-%d")

    snapshot_date_obj = date.fromisoformat(snapshot_date_str)
    if snapshot_date_obj >= expected:
        return (
            False,
            f"snapshot_date={snapshot_date_str} >= expected={expected_str} OK",
            snapshot_date_str,
            expected_str,
        )

    return (
        True,
        (
            f"stale market data: snapshot_date={snapshot_date_str}"
            f" expected={expected_str}"
            f" (run_date={_today.isoformat()})"
        ),
        snapshot_date_str,
        expected_str,
    )


def _compute_startup_equity(state: dict) -> dict[str, Any]:
    """
    Health Check用のcurrent_equity/DD/PEAK_ANOMALYを計算する
    （Broker-as-Sole-SSOT, 2026-07-18）。

    唯一のソース: fetch_broker_snapshot() + compute_live_equity()。
    SignalBridge本体（run_live_signal.py）と完全に同一の関数を使う。
    state ファイルの available_cash/position_qtys・OHLCVキャッシュへの
    フォールバックは行わない（2026-07-15〜17 equity_peak異常値インシデントの
    根本原因はまさにこの独自フォールバック経路が持つ食い違いだった。
    旧実装は filter_live_positions() で正規化した後に LeavesQty キーのみで
    qtyを再抽出しており、LeaveQty/Qty/HoldQty 応答の場合に全ポジションを
    見失うバグを持っていた）。

    broker接続失敗時は current_equity を計算できないため ok=False とし、
    呼び出し側が startup check を失敗させる（LIVE/DRY問わずabort）。

    Returns:
        {equity_peak, current_equity, dd, cash_used, equity_fallback,
         equity_src, broker_available, dd_breach, warnings}
    """
    from src.portfolio.equity import compute_live_equity
    from src.portfolio.broker_source import fetch_broker_snapshot, BrokerSnapshotUnavailable

    ep = float(state.get("equity_peak", 0))

    _broker_fetch_error: str = ""
    _broker_available = False
    current_equity = 0.0
    _cash_used = 0.0
    try:
        from src.kabusapi.client import KabuClient
        _kc = KabuClient()
        _kc.fetch_token()
        _snap = fetch_broker_snapshot(_kc)
        current_equity = compute_live_equity(
            snapshot=_snap, mode="startup", equity_peak=ep, persist_snapshot=False,
        )
        _cash_used = _snap.cash
        _broker_available = True
    except (BrokerSnapshotUnavailable, Exception) as _bf_exc:
        _broker_fetch_error = str(_bf_exc)
        _logger.warning("[EQUITY_PEAK_DIAG] broker snapshot fetch failed: %s", _bf_exc)

    dd = compute_drawdown(current_equity, ep) * 100 if (ep > 0 and _broker_available) else 0.0

    try:
        _state_mtime = datetime.fromtimestamp(
            _PORTFOLIO_FILE.stat().st_mtime, tz=timezone(timedelta(hours=9))
        ).strftime("%Y-%m-%dT%H:%M:%S%z")
    except OSError:
        _state_mtime = "unknown"
    _equity_src = "broker_snapshot" if _broker_available else "unavailable"
    _equity_fallback = not _broker_available
    _logger.info(
        "[EQUITY_PEAK_DIAG] equity_peak=%s current_equity=%s dd=%.2f%% source=%s mtime=%s"
        + (" broker_fetch_error=%s" % _broker_fetch_error if _broker_fetch_error else ""),
        f"{ep:,.0f}", f"{current_equity:,.0f}", dd, _equity_src, _state_mtime,
    )
    if not _broker_available:
        _logger.warning(
            "[DD_WARNING] live equity unavailable — broker snapshot取得失敗のため "
            "current_equity/DDを計算できません。呼び出し側は ok=False として扱うこと。"
        )

    out_warnings: list[str] = []
    dd_breach = ep > 0 and dd < _DD_WARN_THRESHOLD * 100
    if dd_breach:
        _eq_label = f"¥{current_equity:,.0f}(fallback)" if _equity_fallback else f"¥{current_equity:,.0f}"
        out_warnings.append(
            f"⚠️ DD警告: {dd:.1f}%  equity={_eq_label} / peak=¥{ep:,.0f}"
            f"  (BUY_STOP閾値 {_DD_WARN_THRESHOLD*100:.0f}%)"
        )

    if ep > 0 and current_equity > 0:
        _ratio = ep / current_equity
        if _ratio > 1.25:
            _anomaly_src = "(avg_price_fallback)" if _equity_fallback else ""
            out_warnings.append(
                f"⚠️ PEAK_ANOMALY: equity_peak=¥{ep:,.0f}"
                f" / equity=¥{current_equity:,.0f}{_anomaly_src}"
                f" ratio={_ratio:.2f} > 1.25 — state ファイル破損の可能性"
            )

    return {
        "equity_peak":      ep,
        "current_equity":   current_equity,
        "dd":               dd,
        "cash_used":        _cash_used,
        "equity_fallback":  _equity_fallback,
        "equity_src":       _equity_src,
        "broker_available": _broker_available,
        "dd_breach":        dd_breach,
        "warnings":         out_warnings,
    }


# ── メイン公開関数 ────────────────────────────────────────────────────────────

def run_startup_check(is_live: bool = False) -> dict[str, Any]:
    """
    全チェックを実行し結果を返す。

    Args:
        is_live: True → stale data を issues に追加して発注停止。
                 False (DRY) → stale data を warnings に追加して実行継続。

    Returns:
        {
            "ok":                  bool,         # False なら致命的問題あり
            "issues":              list[str],     # 致命的問題リスト
            "warnings":            list[str],     # 警告リスト（発注継続可能）
            "summary":             str,           # 1行要約
            "state":               dict,          # portfolio_state.json の内容（失敗時は{}）
            "stale_data_detected": bool,          # [6] スナップショット陳腐化フラグ
            "snapshot_date":       str | None,    # [6] 実際のスナップショット日付
            "expected_date":       str | None,    # [6] 期待された前営業日
            "api_auth_ok":         bool,          # [7] API トークン取得成否
            "api_auth_msg":        str,           # [7] 認証結果メッセージ
            "timestamp":           str,           # JST チェック時刻
        }
    """
    ts = datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S JST")
    issues:   list[str] = []
    warnings: list[str] = []
    state:    dict[str, Any] = {}

    # [1] API ポート
    api_ok, api_msg = _check_api_port()
    if not api_ok:
        issues.append(api_msg)

    # [7] API 認証プリフライト（[1] 成功時のみ実行）
    auth_ok  = True
    auth_msg = "skipped (port unreachable)"
    if api_ok:
        auth_ok, auth_msg = _check_api_token_with_retry(is_live=is_live)
        if not auth_ok:
            if is_live:
                issues.append(f"[API_AUTH] LIVE発注停止: {auth_msg}")
            else:
                warnings.append(f"[API_AUTH] DRY継続: {auth_msg}")

    # [2][3][5] portfolio_state.json
    state_ok, state_issues, state_warnings, state = _check_portfolio_state()
    issues.extend(state_issues)
    warnings.extend(state_warnings)

    # [4] ロックファイル
    lock_ok, lock_msg = _check_lock_file()
    if not lock_ok:
        issues.append(lock_msg)

    # [6] スナップショット日付（Stale Market Data FAIL-CLOSED）
    stale_detected: bool = False
    snapshot_date:  Optional[str] = None
    expected_date:  Optional[str] = None
    try:
        stale_detected, snap_msg, snapshot_date, expected_date = _check_snapshot_date()
        if stale_detected:
            if is_live:
                issues.append(
                    f"[STALE_DATA] LIVE発注停止: {snap_msg}"
                )
            else:
                warnings.append(
                    f"[STALE_DATA] DRY継続: {snap_msg}"
                )
    except Exception as _snap_err:
        warnings.append(f"[STALE_DATA] チェック例外（スキップ）: {_snap_err}")

    ok = len(issues) == 0

    if ok:
        eq_result = _compute_startup_equity(state)
        ep              = eq_result["equity_peak"]
        current_equity  = eq_result["current_equity"]
        dd              = eq_result["dd"]
        _equity_fallback = eq_result["equity_fallback"]
        _cash_used      = eq_result["cash_used"]
        warnings.extend(eq_result["warnings"])

        # Broker-as-Sole-SSOT (2026-07-18): broker snapshot取得に失敗した場合、
        # current_equity/DDを計算する手段が無い。LIVE/DRY問わずFAIL-CLOSEDとし、
        # state ファイルへの FAIL_OPEN フォールバックは行わない
        # （旧実装のフォールバックが2026-07-15〜17インシデントの一因だった）。
        if not eq_result["broker_available"]:
            issues.append("[BROKER_UNAVAILABLE] broker snapshot取得失敗 — current_equity/DD計算不可")
            ok = False
        else:
            _cb_now = str(state.get("cb_state", "NORMAL"))
            if eq_result["dd_breach"] and _cb_now == "NORMAL":
                warnings.append(f"cb_state=NORMAL だが DD={dd:.1f}% → 手動確認推奨")

    if ok:
        stale_suffix = f" [STALE_DATA: {snapshot_date}→{expected_date}]" if stale_detected else ""
        _dd_display = f"DD={dd:.1f}%" + (" (fallback)" if _equity_fallback else "")
        summary = f"OK — equity_peak={ep:,.0f}円, cash={_cash_used:,.0f}円, {_dd_display}{stale_suffix}"
    else:
        summary = f"FAIL — {len(issues)}件の致命的問題 / {len(warnings)}件の警告"

    return {
        "ok":                  ok,
        "issues":              issues,
        "warnings":            warnings,
        "summary":             summary,
        "state":               state,
        "stale_data_detected": stale_detected,
        "snapshot_date":       snapshot_date,
        "expected_date":       expected_date,
        "api_port_ok":         api_ok,
        "api_auth_ok":         auth_ok,
        "api_auth_msg":        auth_msg,
        "timestamp":           ts,
    }


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    result = run_startup_check()
    print(f"[{result['timestamp']}] {result['summary']}")
    for w in result["warnings"]:
        print(f"  WARNING: {w}")
    for e in result["issues"]:
        print(f"  ERROR:   {e}")
    sys.exit(0 if result["ok"] else 1)
