# scripts/run_ce_pipeline.py
# FIXED: cwd-independent import, absolute error log path, traceback, timestamp

import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # ADDED

# ADDED: cwd-independent import — works from any working directory
_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluate_ce import evaluate
from src.notify import notify

# ADDED: absolute path for error log
_LOGS      = Path(os.environ.get("AI_TRADING_HOME", str(_PROJECT_ROOT))) / "logs"
_ERROR_LOG = _LOGS / "ce_error.log"


def main():
    try:
        result = evaluate()
        notify(result)
    except Exception:
        try:  # FIXED: absolute path + full traceback + timestamp
            _LOGS.mkdir(parents=True, exist_ok=True)
            with open(_ERROR_LOG, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now().isoformat()}]\n{traceback.format_exc()}\n")
        except Exception:
            pass


if __name__ == "__main__":
    main()
