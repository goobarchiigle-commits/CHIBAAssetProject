"""
src/monitoring/watchdog.py
別プロセスで実行するウォッチドッグ。

監視対象:
    1. ログファイルの更新停止（mtime 遅延）
    2. n_trades == 0 継続

動作:
    - 1回目異常 → WARN のみ
    - 連続異常   → monitoring_state.json に halted=True を書き込み KILL

起動例:
    python -m src.monitoring.watchdog
    python -m src.monitoring.watchdog --stale-sec 1800 --interval 60
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

_BASE        = Path(__file__).resolve().parents[2]
_LOG_CSV     = _BASE / "logs" / "pipeline_log.csv"
_STATE_JSON  = _BASE / "runtime" / "monitoring_state.json"

STALE_SEC        = 3_600   # ログ更新停止とみなす閾値（秒）
CHECK_INTERVAL   = 60      # 監視間隔（秒）
ZERO_TRADE_LIMIT = 3       # n_trades==0 が連続 N 回で KILL
WARN_THRESHOLD   = 1       # これ以下は WARN のみ


# ── 内部ユーティリティ ────────────────────────────────────────────────────────────
def _log_age_sec(log_path: Path) -> float | None:
    """ログファイルの最終更新から現在までの経過秒。ファイル不在なら None。"""
    if not log_path.exists():
        return None
    return time.time() - log_path.stat().st_mtime


def _read_n_trades(state_path: Path) -> int | None:
    """monitoring_state.json から n_trades を読む。読めない場合は None。"""
    if not state_path.exists():
        return None
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
        v = data.get("n_trades")
        return int(v) if v is not None else None
    except Exception:
        return None


def _write_kill(state_path: Path, reason: str) -> None:
    """monitoring_state.json に halted=True を書き込む。"""
    try:
        data = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    except Exception:
        data = {}

    data.update({
        "halted":            True,
        "mode":              "HALT",
        "halt_reason":       f"watchdog: {reason}",
        "halt_ts":           time.time(),
        "global_multiplier": 0.0,
    })
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[WATCHDOG] KILL: halted=True を書き込みました ({reason})", flush=True)


# ── メインループ ──────────────────────────────────────────────────────────────────
def run_watchdog(
    log_path:       Path = _LOG_CSV,
    state_path:     Path = _STATE_JSON,
    stale_sec:      int  = STALE_SEC,
    check_interval: int  = CHECK_INTERVAL,
    zero_limit:     int  = ZERO_TRADE_LIMIT,
    warn_threshold: int  = WARN_THRESHOLD,
) -> None:
    print(
        f"[WATCHDOG] 起動  log={log_path}  state={state_path}\n"
        f"           stale_sec={stale_sec}  interval={check_interval}s  zero_limit={zero_limit}",
        flush=True,
    )

    consecutive_stale = 0
    consecutive_zero  = 0

    while True:
        # ── ログ更新停止チェック ────────────────────────────────────────────────────
        age = _log_age_sec(log_path)
        if age is None:
            consecutive_stale += 1
            msg = f"[WATCHDOG] STALE: ログ不在 {log_path} (連続 {consecutive_stale}回)"
        elif age > stale_sec:
            consecutive_stale += 1
            msg = f"[WATCHDOG] STALE: 更新停止 {age:.0f}秒経過 (連続 {consecutive_stale}回)"
        else:
            consecutive_stale = 0
            msg = None

        if msg:
            if consecutive_stale <= warn_threshold:
                print(msg + " → WARN", flush=True)
            else:
                print(msg + " → KILL", flush=True)
                _write_kill(state_path, f"STALE {age:.0f}s")

        # ── n_trades == 0 継続チェック ─────────────────────────────────────────────
        n = _read_n_trades(state_path)
        if n is not None and n == 0:
            consecutive_zero += 1
            zmsg = f"[WATCHDOG] ZERO_TRADE: n_trades==0 継続 {consecutive_zero}回"
            if consecutive_zero <= warn_threshold:
                print(zmsg + " → WARN", flush=True)
            else:
                print(zmsg + " → KILL", flush=True)
                _write_kill(state_path, "ZERO_TRADE")
        else:
            consecutive_zero = 0

        time.sleep(check_interval)


# ── CLI エントリーポイント ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitoring Watchdog")
    parser.add_argument("--log",        type=Path, default=_LOG_CSV,      help="監視対象ログ CSV")
    parser.add_argument("--state",      type=Path, default=_STATE_JSON,   help="monitoring_state.json パス")
    parser.add_argument("--stale-sec",  type=int,  default=STALE_SEC,     help="更新停止とみなす閾値（秒）")
    parser.add_argument("--interval",   type=int,  default=CHECK_INTERVAL, help="チェック間隔（秒）")
    parser.add_argument("--zero-limit", type=int,  default=ZERO_TRADE_LIMIT, help="n_trades==0 連続 KILL 閾値")
    args = parser.parse_args()

    run_watchdog(
        log_path       = args.log,
        state_path     = args.state,
        stale_sec      = args.stale_sec,
        check_interval = args.interval,
        zero_limit     = args.zero_limit,
    )
