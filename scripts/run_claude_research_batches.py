from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"

SCRIPTS = [
    REPO_ROOT / "scripts" / "run_filter_ablation_batch.py",
    REPO_ROOT / "scripts" / "run_exit_sensitivity_batch.py",
    REPO_ROOT / "scripts" / "run_regime_decomposition_2025.py",
]
OUTPUT_TXT = RESULTS_DIR / "research_batch_best_combination.txt"


def main() -> int:
    for script in SCRIPTS:
        print(f"[SCENARIO] running {script.name}")
        completed = subprocess.run([sys.executable, str(script)], cwd=str(REPO_ROOT), check=False)
        if completed.returncode != 0:
            return completed.returncode

    filter_df = pd.read_csv(RESULTS_DIR / "filter_ablation_report.csv")
    exit_df = pd.read_csv(RESULTS_DIR / "exit_sensitivity_grid.csv")
    regime_df = pd.read_csv(RESULTS_DIR / "oos_2025_regime_report.csv")

    best_filter = filter_df.sort_values(["sharpe", "expectancy", "trade_count"], ascending=[False, False, False]).iloc[0]
    best_exit = exit_df.sort_values(["sharpe", "expectancy", "trade_count"], ascending=[False, False, False]).iloc[0]
    worst_regime = regime_df.sort_values(["sharpe", "expectancy", "trade_count"], ascending=[True, True, True]).iloc[0]

    lines = [
        "Best Combination Candidate",
        f"filter={best_filter['scenario_name']}",
        f"exit_tuning={best_exit['parameter_name']}={best_exit['parameter_value']}",
        f"worst_regime_to_watch={worst_regime['regime']}",
        f"candidate_note=Use the best filter-ablation winner, apply the best single exit tweak, and monitor the worst regime first.",
    ]
    OUTPUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[SCENARIO] best combination written to {OUTPUT_TXT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
