from __future__ import annotations

import time
from itertools import combinations
from pathlib import Path

import pandas as pd

from research_batch_utils import (
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


OUTPUT_CSV = RESULTS_DIR / "filter_ablation_report.csv"
OUTPUT_TXT = RESULTS_DIR / "filter_ablation_summary.txt"


def build_base_scenarios() -> list[dict]:
    return [
        {
            "scenario_name": "baseline",
            "enable_filters": False,
            "enable_volatility_filter": False,
            "enable_atr_filter": False,
            "enable_volume_filter": False,
            "enable_market_filter": False,
        },
        {
            "scenario_name": "all_filter",
            "enable_filters": True,
            "enable_volatility_filter": True,
            "enable_atr_filter": True,
            "enable_volume_filter": True,
            "enable_market_filter": True,
        },
        {
            "scenario_name": "A1_vol_filter_off",
            "enable_filters": True,
            "enable_volatility_filter": False,
            "enable_atr_filter": True,
            "enable_volume_filter": True,
            "enable_market_filter": True,
        },
        {
            "scenario_name": "A2_regime_filter_off",
            "enable_filters": True,
            "enable_volatility_filter": True,
            "enable_atr_filter": True,
            "enable_volume_filter": True,
            "enable_market_filter": False,
        },
        {
            "scenario_name": "A3_atr_filter_off",
            "enable_filters": True,
            "enable_volatility_filter": True,
            "enable_atr_filter": False,
            "enable_volume_filter": True,
            "enable_market_filter": True,
        },
        {
            "scenario_name": "A4_volume_ma_off",
            "enable_filters": True,
            "enable_volatility_filter": True,
            "enable_atr_filter": True,
            "enable_volume_filter": False,
            "enable_market_filter": True,
        },
    ]


def build_pairwise_ablation_scenarios() -> list[dict]:
    toggles = {
        "vol": "enable_volatility_filter",
        "regime": "enable_market_filter",
        "atr": "enable_atr_filter",
        "volume": "enable_volume_filter",
    }
    scenarios: list[dict] = []
    for left, right in combinations(toggles.items(), 2):
        scenario = {
            "scenario_name": f"pair_{left[0]}_{right[0]}_off",
            "enable_filters": True,
            "enable_volatility_filter": True,
            "enable_atr_filter": True,
            "enable_volume_filter": True,
            "enable_market_filter": True,
        }
        scenario[left[1]] = False
        scenario[right[1]] = False
        scenarios.append(scenario)
    return scenarios


def main() -> int:
    started_at = time.perf_counter()
    cfg = build_cfg()
    dataset = load_oos_dataset()
    scenarios = build_base_scenarios()

    rows: list[dict] = []
    total = len(scenarios)
    baseline_sharpe = None

    for index, scenario in enumerate(scenarios, start=1):
        run_started = time.perf_counter()
        kwargs = {k: v for k, v in scenario.items() if k != "scenario_name"}
        result = run_scenario_oos(dataset, cfg, scenario_name=scenario["scenario_name"], **kwargs)
        scenario_log(scenario["scenario_name"], index, total, result, run_started, baseline_sharpe=baseline_sharpe)
        row = summarize_result(result, scenario_name=scenario["scenario_name"])
        rows.append(row)
        if scenario["scenario_name"] == "baseline":
            baseline_sharpe = row["sharpe"]

    df = pd.DataFrame(rows)
    baseline = df.loc[df["scenario_name"] == "baseline"].iloc[0]
    df["delta_sharpe_vs_baseline"] = (df["sharpe"] - float(baseline["sharpe"])).round(3)
    df["delta_trade_count_vs_baseline"] = (df["trade_count"] - int(baseline["trade_count"])).astype(int)
    ordered = [
        "scenario_name",
        "trade_count",
        "hit_rate",
        "expectancy",
        "profit_factor",
        "sharpe",
        "avg_hold_days",
        "avg_win",
        "avg_loss",
        "count_win",
        "count_loss",
        "exit_rsr_pct",
        "exit_atr_pct",
        "exit_turtle_pct",
        "delta_sharpe_vs_baseline",
        "delta_trade_count_vs_baseline",
    ]
    df = df[ordered].sort_values(["sharpe", "expectancy", "trade_count"], ascending=[False, False, False])
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    best = best_row(df)
    summary_lines = [
        "Filter Ablation Summary",
        f"best_scenario={best['scenario_name']}",
        f"best_sharpe={float(best['sharpe']):.3f}",
        f"best_expectancy={float(best['expectancy']):.3f}",
        f"baseline_sharpe={float(baseline['sharpe']):.3f}",
        f"baseline_trade_count={int(baseline['trade_count'])}",
        f"best_delta_sharpe_vs_baseline={float(best['delta_sharpe_vs_baseline']):+.3f}",
        f"best_delta_trade_count_vs_baseline={int(best['delta_trade_count_vs_baseline']):+d}",
        f"pairwise_extension_ready={len(build_pairwise_ablation_scenarios())} scenarios",
        f"elapsed_total_sec={time.perf_counter() - started_at:.2f}",
    ]
    summary_lines.extend(report_footer(scenario_count=len(df), best_sharpe=float(best["sharpe"]), baseline_sharpe=float(baseline["sharpe"])))
    write_text(OUTPUT_TXT, summary_lines)

    print(f"[SCENARIO] completed filter ablation scenarios={len(df)} best={best['scenario_name']}")
    print(f"[ELAPSED] total: {time.perf_counter() - started_at:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
