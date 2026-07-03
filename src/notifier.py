"""
src/notifier.py
エラー・完了通知モジュール。Gmail SMTP 経由でメール送信。

設定 (.env):
    NOTIFY_SMTP_USER     = your@gmail.com
    NOTIFY_SMTP_PASSWORD = xxxx xxxx xxxx xxxx  ← Googleアカウント > アプリパスワード
    NOTIFY_TO            = destination@example.com
    NOTIFY_SMTP_HOST     = smtp.gmail.com  (省略可)
    NOTIFY_SMTP_PORT     = 587             (省略可)

SMTP未設定時: メール不送信、logs/notifications.log へのフォールバックのみ。
スレッド: daemon thread で fire-and-forget（メール失敗でもメインプロセスをブロックしない）
"""
from __future__ import annotations

import logging
import os
from dotenv import load_dotenv
import smtplib
import threading
import traceback
from datetime import datetime, timezone, timedelta
from email.mime.text import MIMEText
from pathlib import Path

JST = timezone(timedelta(hours=9))
logger = logging.getLogger(__name__)

# ── パス ────────────────────────────────────────────────────────────────────
_HERE      = Path(__file__).resolve().parent
load_dotenv(_HERE / ".env")
_BASE_DIR  = Path(os.environ.get("AI_TRADING_HOME", str(_HERE.parent)))
_NOTIFY_LOG = _BASE_DIR / "logs" / "notifications.log"


def _get_cfg() -> dict[str, str]:
    """環境変数から通知設定を取得する。未設定キーは空文字。"""
    return {
        "host":     os.environ.get("NOTIFY_SMTP_HOST",     "smtp.gmail.com"),
        "port":     os.environ.get("NOTIFY_SMTP_PORT",     "587"),
        "user":     os.environ.get("NOTIFY_SMTP_USER",     ""),
        "password": os.environ.get("NOTIFY_SMTP_PASSWORD", ""),
        "to":       os.environ.get("NOTIFY_TO",            ""),
    }


def _fallback_log(subject: str, body: str) -> None:
    """メール送信失敗時のフォールバック: ファイルへ記録。"""
    try:
        _NOTIFY_LOG.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S JST")
        entry = f"[{ts}] {subject}\n{body}\n{'─'*60}\n"
        with _NOTIFY_LOG.open("a", encoding="utf-8") as f:
            f.write(entry)
    except Exception:
        pass


def _send_email(subject: str, body: str) -> bool:
    """
    Gmail SMTP でメール送信する（同期）。
    成功: True / 失敗 or 設定なし: False
    """
    cfg = _get_cfg()
    if not cfg["user"] or not cfg["password"] or not cfg["to"]:
        return False

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"]    = cfg["user"]
    msg["To"]      = cfg["to"]
    msg["Date"]    = datetime.now(JST).strftime("%a, %d %b %Y %H:%M:%S %z")

    try:
        with smtplib.SMTP(cfg["host"], int(cfg["port"]), timeout=15) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.login(cfg["user"], cfg["password"])
            smtp.sendmail(cfg["user"], [cfg["to"]], msg.as_string())
        return True
    except Exception:
        logger.warning("メール送信失敗: %s", traceback.format_exc(limit=3))
        return False


def _notify_async(subject: str, body: str) -> None:
    """daemon thread で非同期送信し、失敗時はファイルにフォールバック。"""
    import time

    def _run() -> None:
        t0 = time.monotonic()
        cfg = _get_cfg()
        recipient = cfg.get("to", "")
        ok = _send_email(subject, body)
        elapsed = time.monotonic() - t0
        if ok:
            logger.info(
                "[NOTIFY_SUCCESS] recipient=%s subject=%s elapsed=%.2fs",
                recipient, subject, elapsed,
            )
        else:
            logger.warning(
                "[NOTIFY_FAIL] recipient=%s subject=%s elapsed=%.2fs",
                recipient, subject, elapsed,
            )
            _fallback_log(subject, body)

    t = threading.Thread(target=_run, daemon=True)
    t.start()


# ── 公開 API ─────────────────────────────────────────────────────────────────

def notify_success(body: str, subject_suffix: str = "") -> None:
    """✅ 発注完了・正常終了通知。"""
    suffix = f" {subject_suffix}" if subject_suffix else ""
    _notify_async(f"✅ CHIBA 発注完了{suffix}", body)


def notify_warning(body: str, subject_suffix: str = "") -> None:
    """⚠️ 警告通知（処理は継続）。"""
    suffix = f" {subject_suffix}" if subject_suffix else ""
    _notify_async(f"⚠️ CHIBA 警告{suffix}", body)


def notify_error(body: str, subject_suffix: str = "") -> None:
    """❌ エラー通知（処理停止）。"""
    suffix = f" {subject_suffix}" if subject_suffix else ""
    _notify_async(f"❌ CHIBA エラー{suffix}", body)


def notify_dry_run(body: str) -> None:
    """📋 ドライラン結果通知。"""
    _notify_async("📋 CHIBA ドライラン結果", body)


def wait_pending(timeout: float = 10.0) -> None:
    """
    非同期スレッドの完了を最大 timeout 秒待つ。
    プロセス終了前に呼ぶと通知が確実に届く。
    """
    import time
    deadline = time.monotonic() + timeout
    for t in threading.enumerate():
        if t.daemon and t.name.startswith("Thread"):
            remaining = deadline - time.monotonic()
            if remaining > 0:
                t.join(timeout=remaining)
