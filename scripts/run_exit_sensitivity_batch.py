from __future__ import annotations

import time
from dataclasses import replace

import pandas as pd

from research_batch_utils import (
    CURRENT_RSR_EXIT_THRESHOLD,
    CURRENT_TRAIL_ATR_MULT,
    RESULTS_DIR,
    best_row,
    build_cfg,
    load_oos_dataset,
    report_footer,
    run_scenario_oos,
    scenario_log,
    summarize_result,
    write_text,
)


OUTPUT_CSV = RESULTS_DIR / "exit_sensitivity_grid.csv"
OUTPUT_TXT = RESULTS_DIR / "exit_best_config.txt"

ATR_MULT_VALUES = [1.0, 1.5, 2.0, 2.5, 3.0]
RSR_EXIT_DELTAS = [-5, -10, -15, -20]
MAX_HOLD_VALUES = [None, 20, 15, 10, 7]


def cfg_with_max_hold(max_hold_days):
    cfg = build_cfg()
    return replace(cfg, risk=replace(cfg.risk, max_hold_days=max_hold_days))


def main() -> int:
    started_at = time.perf_counter()
    dataset = load_oos_dataset()
    current_cfg = build_cfg()
    baseline_cfg = build_cfg()

    baseline_result = run_scenario_oos(
        dataset,
        baseline_cfg,
        scenario_name="baseline_filters_off",
        enable_filters=False,
        enable_volatility_filter=False,
        enable_atr_filter=False,
        enable_volume_filter=False,
        enable_market_filter=False,
        rsr_exit_threshold=CURRENT_RSR_EXIT_THRESHOLD,
        trail_atr_mult=CURRENT_TRAIL_ATR_MULT,
    )
    current_result = run_scenario_oos(
        dataset,
        current_cfg,
        scenario_name="current",
        rsr_exit_threshold=CURRENT_RSR_EXIT_THRESHOLD,
        trail_atr_mult=CURRENT_TRAIL_ATR_MULT,
    )
    baseline_summary = summarize_result(baseline_result, scenario_name="baseline")
    current_summary = summarize_result(current_result, scenario_name="current")
    baseline_sharpe = float(baseline_summary["sharpe"])
    current_sharpe = float(current_summary["sharpe"])

    rows: list[dict] = []
    scenarios: list[tuple[str, str, object]] = []
    scenarios.extend(("ATR trail", "atr_mult", value) for value in ATR_MULT_VALUES)
    scenarios.extend(("RSR exit", "rsr_exit_delta", value) for value in RSR_EXIT_DELTAS)
    scenarios.extend(("Max hold", "max_hold_days", value) for value in MAX_HOLD_VALUES)

    for index, (test_type, parameter_name, parameter_value) in enumerate(scenarios, start=1):
        run_started = time.perf_counter()
        if parameter_name == "atr_mult":
            cfg = current_cfg
            result = run_scenario_oos(
                dataset,
                cfg,
                scenario_name=f"atr_mult_{parameter_value}",
                rsr_exit_threshold=CURRENT_RSR_EXIT_THRESHOLD,
                trail_atr_mult=float(parameter_value),
            )
        elif parameter_name == "rsr_exit_delta":
            cfg = current_cfg
            threshold = CURRENT_RSR_EXIT_THRESHOLD + float(parameter_value)
            result = run_scenario_oos(
                dataset,
                cfg,
                scenario_name=f"rsr_exit_delta_{parameter_value}",
                rsr_exit_threshold=threshold,
                trail_atr_mult=CURRENT_TRAIL_ATR_MULT,
            )
        else:
            cfg = cfg_with_max_hold(parameter_value)
            result = run_scenario_oos(
                dataset,
                cfg,
                scenario_name=f"max_hold_{parameter_value}",
                rsr_exit_threshold=CURRENT_RSR_EXIT_THRESHOLD,
                trail_atr_mult=CURRENT_TRAIL_ATR_MULT,
            )

        scenario_log(str(parameter_name) + "=" + str(parameter_value), index, len(scenarios), result, run_started, baseline_sharpe=baseline_sharpe)
        row = summarize_result(result, scenario_name=f"{parameter_name}={parameter_value}")
        row.update({
            "test_type": test_type,
            "parameter_name": parameter_name,
            "parameter_value": parameter_value,
            "delta_vs_current": round(float(row["sharpe"]) - current_sharpe, 3),
            "delta_vs_baseline": round(float(row["sharpe"]) - baseline_sharpe, 3),
        })
        rows.append(row)

    df = pd.DataFrame(rows)
    ordered = [
        "test_type",
        "parameter_name",
        "parameter_value",
        "trade_count",
        "hit_rate",
        "expectancy",
        "profit_factor",
        "sharpe",
        "avg_hold_days",
        "max_drawdown",
        "exit_reason_distribution",
        "delta_vs_current",
        "delta_vs_baseline",
    ]
    df = df[ordered].sort_values(["sharpe", "expectancy", "trade_count"], ascending=[False, False, False])
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    best = best_row(df)
    summary_lines = [
        "Exit Sensitivity Best Config",
        f"best_test_type={best['test_type']}",
        f"best_parameter={best['parameter_name']}",
        f"best_value={best['parameter_value']}",
        f"best_sharpe={float(best['sharpe']):.3f}",
        f"best_delta_vs_current={float(best['delta_vs_current']):+.3f}",
        f"best_delta_vs_baseline={float(best['delta_vs_baseline']):+.3f}",
        f"current_sharpe={current_sharpe:.3f}",
        f"baseline_sharpe={baseline_sharpe:.3f}",
        f"elapsed_total_sec={time.perf_counter() - started_at:.2f}",
    ]
    summary_lines.extend(report_footer(scenario_count=len(df), best_sharpe=float(best["sharpe"]), baseline_sharpe=baseline_sharpe))
    write_text(OUTPUT_TXT, summary_lines)

    print(f"[SCENARIO] current sharpe={current_sharpe:.3f} baseline sharpe={baseline_sharpe:.3f}")
    print(f"[SCENARIO] completed exit sensitivity scenarios={len(df)} best={best['parameter_name']}={best['parameter_value']}")
    print(f"[ELAPSED] total: {time.perf_counter() - started_at:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
