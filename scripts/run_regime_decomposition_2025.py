from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.utils.memory import collect_and_log_process_memory

from research_batch_utils import (
    CURRENT_RSR_EXIT_THRESHOLD,
    CURRENT_TRAIL_ATR_MULT,
    OOS_END,
    OOS_START,
    RESULTS_DIR,
    build_cfg,
    load_oos_dataset,
    report_footer,
    run_scenario_oos,
    summarize_result,
    try_load_usdjpy,
    write_text,
)


OUTPUT_CSV = RESULTS_DIR / "oos_2025_regime_report.csv"
OUTPUT_TXT = RESULTS_DIR / "oos_2025_regime_summary.txt"

TOPIX_RISING_THRESHOLD = 0.05
TOPIX_FALLING_THRESHOLD = -0.05
TOPIX_RANGE_THRESHOLD = 0.03
USDJPY_STRONG_THRESHOLD = 145.0


def build_regime_frame(topix_close: pd.Series, usdjpy_close: pd.Series) -> tuple[pd.DataFrame, str | None]:
    idx = pd.date_range(OOS_START, OOS_END, freq="B")
    topix = topix_close.reindex(idx).ffill()
    topix_ret20 = topix.pct_change(20)
    topix_vol20 = topix.pct_change().rolling(20, min_periods=10).std() * np.sqrt(252)
    vol_threshold = float(topix_vol20.dropna().quantile(0.75)) if not topix_vol20.dropna().empty else np.inf

    usd = usdjpy_close.reindex(idx).ffill() if not usdjpy_close.empty else pd.Series(index=idx, dtype=float)
    note = None if not usd.dropna().empty else "USDJPY data unavailable; YEN_STRONG days set to 0."

    regime = pd.DataFrame(index=idx)
    regime["RISING"] = topix_ret20 >= TOPIX_RISING_THRESHOLD
    regime["FALLING"] = topix_ret20 <= TOPIX_FALLING_THRESHOLD
    regime["RANGE"] = topix_ret20.abs() <= TOPIX_RANGE_THRESHOLD
    regime["HIGH_VOL"] = topix_vol20 >= vol_threshold
    regime["YEN_STRONG"] = usd <= USDJPY_STRONG_THRESHOLD if not usd.dropna().empty else False
    return regime.fillna(False), note


def enrich_trade_dates(result: dict) -> pd.DataFrame:
    trades = pd.DataFrame(result.get("_trades", []))
    if trades.empty:
        return trades
    eq_index = result["equity_curve"].index
    trades = trades.copy()
    trades["entry_date"] = trades["entry_idx"].apply(lambda idx: eq_index[int(idx)] if 0 <= int(idx) < len(eq_index) else pd.NaT)
    trades["exit_date"] = trades["exit_idx"].apply(lambda idx: eq_index[int(idx)] if 0 <= int(idx) < len(eq_index) else pd.NaT)
    trades["ret_pct"] = (trades["exit"] / trades["entry"] - 1.0) * 100.0
    return trades


def regime_metrics(regime_name: str, mask: pd.Series, result: dict, trades: pd.DataFrame) -> dict:
    eq = result["equity_curve"].reindex(mask.index).ffill()
    regime_eq = eq[mask]
    regime_dr = regime_eq.pct_change().dropna()
    sharpe = float(regime_dr.mean() / regime_dr.std() * np.sqrt(252)) if len(regime_dr) > 1 and regime_dr.std() > 0 else 0.0

    if trades.empty:
        selected = trades
    else:
        selected = trades[trades["entry_date"].isin(mask.index[mask])]

    trade_count = int(len(selected))
    hit_rate = float((selected["pnl"] > 0).mean() * 100.0) if trade_count else 0.0
    avg_win = float(selected.loc[selected["pnl"] > 0, "ret_pct"].mean()) if trade_count else 0.0
    avg_loss = float(selected.loc[selected["pnl"] <= 0, "ret_pct"].mean()) if trade_count else 0.0
    expectancy = (hit_rate / 100.0) * avg_win + (1.0 - hit_rate / 100.0) * avg_loss if trade_count else 0.0
    avg_hold = float((selected["exit_idx"] - selected["entry_idx"]).mean()) if trade_count else 0.0

    reasons = selected["reason"].value_counts().to_dict() if trade_count else {}
    total = max(1, trade_count)
    exit_rsr = float(reasons.get("RSR_EXIT", 0) / total * 100.0)
    exit_atr = float((reasons.get("ATR_TRAIL", 0) + reasons.get("TRAIL_EXIT", 0)) / total * 100.0)
    exit_turtle = float(reasons.get("STRATEGY_EXIT", 0) / total * 100.0)

    return {
        "regime": regime_name,
        "days": int(mask.sum()),
        "trade_count": trade_count,
        "hit_rate": round(hit_rate, 2),
        "expectancy": round(expectancy, 3),
        "sharpe": round(sharpe, 3),
        "avg_hold_days": round(avg_hold, 2),
        "exit_rsr_pct": round(exit_rsr, 2),
        "exit_atr_pct": round(exit_atr, 2),
        "exit_turtle_pct": round(exit_turtle, 2),
    }


def main() -> int:
    started_at = time.perf_counter()
    dataset = load_oos_dataset()
    cfg = build_cfg()
    result = run_scenario_oos(
        dataset,
        cfg,
        scenario_name="regime_base",
        rsr_exit_threshold=CURRENT_RSR_EXIT_THRESHOLD,
        trail_atr_mult=CURRENT_TRAIL_ATR_MULT,
    )
    overall = summarize_result(result, scenario_name="overall")
    topix_close = dataset[6]
    usdjpy_close = try_load_usdjpy()
    regime_df, note = build_regime_frame(topix_close, usdjpy_close)
    trades = enrich_trade_dates(result)

    rows = [regime_metrics(name, regime_df[name], result, trades) for name in ["RISING", "FALLING", "RANGE", "HIGH_VOL", "YEN_STRONG"]]
    df = pd.DataFrame(rows).sort_values(["sharpe", "expectancy", "trade_count"], ascending=[True, True, True])
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    worst = df.iloc[0]
    best = df.sort_values(["sharpe", "expectancy", "trade_count"], ascending=[False, False, False]).iloc[0]
    summary_lines = [
        "OOS 2025 Regime Summary",
        f"overall_sharpe={float(overall['sharpe']):.3f}",
        f"overall_trade_count={int(overall['trade_count'])}",
        f"worst_regime={worst['regime']}",
        f"worst_regime_sharpe={float(worst['sharpe']):.3f}",
        f"worst_regime_expectancy={float(worst['expectancy']):.3f}",
        f"worst_regime_delta_vs_overall={float(worst['sharpe']) - float(overall['sharpe']):+.3f}",
        f"best_regime={best['regime']}",
        f"best_regime_sharpe={float(best['sharpe']):.3f}",
        f"usd_jpy_threshold={USDJPY_STRONG_THRESHOLD:.1f}",
        f"elapsed_total_sec={time.perf_counter() - started_at:.2f}",
    ]
    if note is not None:
        summary_lines.append(note)
    summary_lines.extend(report_footer(scenario_count=len(df), best_sharpe=float(best["sharpe"]), baseline_sharpe=float(overall["sharpe"])))
    write_text(OUTPUT_TXT, summary_lines)

    for row in rows:
        print(f"[SCENARIO] {row['regime']} days={row['days']} trades={row['trade_count']} sharpe={row['sharpe']:.3f} baseline_delta={row['sharpe'] - float(overall['sharpe']):+.3f}")
    collect_and_log_process_memory("regime_decomposition")
    print(f"[ELAPSED] total: {time.perf_counter() - started_at:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
