# src/notify.py
# FIXED: cwd-independent absolute paths, exception handling, duplicate send guard

import os
import sys
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # ADDED

_HERE        = Path(__file__).resolve().parent
_LOGS        = Path(os.environ.get("AI_TRADING_HOME", str(_HERE.parent))) / "logs"  # FIXED: absolute
_RESULT_FILE = _LOGS / "ce_eval_result.txt"
_GUARD_FILE  = _LOGS / "ce_notify_sent.txt"  # ADDED: duplicate guard


def notify(result):
    msg = f"""
[CE Evaluation]

time: {datetime.now().isoformat()}
days: {result.get('days', 0)}

--- Quality ---
cum_delta: {result.get('cum_delta', 0):.4f}
sign_match: {result.get('sign_match', 0):.3f}
trade_sign: {result.get('trade_sign_match', 0):.3f}
corr: {result.get('corr', 0):.3f}
tstat: {result.get('tstat', 0):.2f}

--- Decision ---
decision: {result.get('decision', 'HOLD')}
finalized: {result.get('finalized', False)}

--- Data ---
issues: {result.get('data_issues', [])}
"""

    print(msg)

    try:  # ADDED: swallow all write errors
        _LOGS.mkdir(parents=True, exist_ok=True)  # FIXED: absolute path

        # ADDED: same-day duplicate guard
        today = datetime.now().strftime("%Y-%m-%d")
        if _GUARD_FILE.exists() and _GUARD_FILE.read_text(encoding="utf-8").strip() == today:
            return

        with open(_RESULT_FILE, "a", encoding="utf-8") as f:  # FIXED: absolute path
            f.write(msg + "\n")

        _GUARD_FILE.write_text(today, encoding="utf-8")  # ADDED: mark sent
    except Exception:
        pass
