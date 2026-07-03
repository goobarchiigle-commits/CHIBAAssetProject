"""
tools/setup_exit_intelligence_scheduler.py — Windows Task Scheduler setup

Registers a daily Task Scheduler job that runs exit intelligence report
every trading day at 16:00 JST (after market close).

Usage:
    python tools/setup_exit_intelligence_scheduler.py [--remove]

Requires: Windows, administrator privileges (for schtasks /create).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

TASK_NAME = "CHIBAAssetProject_ExitIntelligenceReport"
SCRIPT_PATH = Path(__file__).resolve().parent.parent / "src" / "analytics" / "exit_intelligence" / "daily_report.py"


def _python_exe() -> str:
    return sys.executable


def _create_task(base_dir: Path) -> None:
    python = _python_exe()
    script = str(SCRIPT_PATH)
    log_dir = base_dir / "logs" / "exit_intelligence"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "daily_report.log"

    # PowerShell command for the action
    action_cmd = (
        f'cmd /c "{python}" "{script}" >> "{log_file}" 2>&1'
    )

    # schtasks: weekdays at 16:00 (JST = system time if system is JST)
    cmd = [
        "schtasks", "/create",
        "/tn", TASK_NAME,
        "/tr", action_cmd,
        "/sc", "WEEKLY",
        "/d", "MON,TUE,WED,THU,FRI",
        "/st", "16:00",
        "/f",   # force overwrite
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"[OK] Task registered: {TASK_NAME}")
        print(f"     Schedule: Weekdays 16:00")
        print(f"     Log: {log_file}")
    else:
        print(f"[ERROR] schtasks failed:\n{result.stderr}")
        sys.exit(1)


def _remove_task() -> None:
    cmd = ["schtasks", "/delete", "/tn", TASK_NAME, "/f"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"[OK] Task removed: {TASK_NAME}")
    else:
        print(f"[WARN] Could not remove: {result.stderr}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Setup exit intelligence Task Scheduler job")
    parser.add_argument("--remove", action="store_true", help="Remove the task instead of creating it")
    parser.add_argument("--base-dir", default=os.environ.get("AI_TRADING_HOME", "C:/ai-trading"),
                        help="Project base directory")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)

    if args.remove:
        _remove_task()
    else:
        _create_task(base_dir)


if __name__ == "__main__":
    main()
