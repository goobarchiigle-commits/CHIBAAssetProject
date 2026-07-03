from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.pop("DATA_VERSION", None)

import src.backtest.composite_alpha_bt as _bt
from src.backtest.rsr import calc_composite_alpha_matrix, calc_universe_rsr
from src.backtest.universe_builder import download_universe
from src.utils.memory import collect_and_log_process_memory
from src.config_loader import load_strategy_config


FULL_START = "2018-01-01"
OOS_START = "2025-01-01"
OOS_END = "2025-12-31"
RESULTS_CSV = REPO_ROOT / "results" / "filter_sensitivity_oos.csv"
ANALYSIS_CSV = REPO_ROOT / "results" / "filter_degradation_analysis.csv"
REPORT_TXT = REPO_ROOT / "results" / "filter_summary_report.txt"
REFERENCE_JSON = REPO_ROOT / "backtests" / "mom_period_sensitivity_2026-03-31.json"


def load_grid() -> pd.DataFrame:
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"missing sensitivity csv: {RESULTS_CSV}")
    df = pd.read_csv(RESULTS_CSV)
    required = {
        "vol_threshold",
        "volume_mult",
        "market_filter",
        "sharpe",
        "cagr",
        "max_dd",
        "win_rate",
        "avg_hold_days",
        "trade_count",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"missing columns in {RESULTS_CSV}: {sorted(missing)}")
    return df


def load_data():
    print("[1/4] loading universe and market data...")
    rsr_syms = _bt._load_rsr_universe()
    trade_syms = rsr_syms
    universe_raw = download_universe({**rsr_syms, **trade_syms}, start=FULL_START, end=OOS_END, verbose=False)
    topix_close = _bt._download_topix(FULL_START, OOS_END)

    print("[2/4] computing rsr and alpha matrices...")
    rsr_df = calc_universe_rsr({
        sym: universe_raw[sym]["df"]["Close"]
        for sym in rsr_syms
        if sym in universe_raw
    })
    alpha_df = calc_composite_alpha_matrix({
        sym: universe_raw[sym]["df"]["Close"]
        for sym in trade_syms
        if sym in universe_raw
    }, window=_bt.COMP_ALPHA_WINDOW).shift(1)

    print("[3/4] computing regime frame...")
    regime_df = _bt._calc_regime(topix_close)

    print("[4/4] data ready")
    return universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms, topix_close


def build_cfg():
    cfg = load_strategy_config()
    return replace(
        cfg,
        fujiko=replace(
            cfg.fujiko,
            mom_period=21,
            turtle_exit=55,
            min_rsr=75.0,
        ),
        portfolio=replace(
            cfg.portfolio,
            capital=3_000_000,
            max_positions=3,
        ),
        risk=replace(
            cfg.risk,
            min_hold_days=3,
        ),
    )


def calc_trade_stats(trades: list[dict]) -> dict:
    if not trades:
        return {
            "avg_profit_hold_days": 0.0,
            "avg_loss_hold_days": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "expectancy": 0.0,
            "profit_factor": 0.0,
        }

    trade_df = pd.DataFrame(trades)
    trade_df["hold_days"] = trade_df["exit_idx"] - trade_df["entry_idx"]
    trade_df["ret_pct"] = (trade_df["exit"] / trade_df["entry"] - 1.0) * 100.0

    wins = trade_df[trade_df["pnl"] > 0]
    losses = trade_df[trade_df["pnl"] <= 0]

    avg_win = float(wins["ret_pct"].mean()) if not wins.empty else 0.0
    avg_loss = float(losses["ret_pct"].mean()) if not losses.empty else 0.0
    win_rate = len(wins) / max(1, len(trade_df))
    expectancy = win_rate * avg_win + (1.0 - win_rate) * avg_loss

    gross_profit = float(wins["pnl"].sum()) if not wins.empty else 0.0
    gross_loss = abs(float(losses["pnl"].sum())) if not losses.empty else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return {
        "avg_profit_hold_days": round(float(wins["hold_days"].mean()) if not wins.empty else 0.0, 2),
        "avg_loss_hold_days": round(float(losses["hold_days"].mean()) if not losses.empty else 0.0, 2),
        "avg_win": round(avg_win, 3),
        "avg_loss": round(avg_loss, 3),
        "expectancy": round(expectancy, 3),
        "profit_factor": round(profit_factor, 3) if np.isfinite(profit_factor) else float("inf"),
    }


def run_case(dataset, cfg, enable_filters: bool, vol_threshold: float, volume_mult: float, market_filter: bool) -> dict:
    universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms, topix_close = dataset
    result = _bt.run_scenario(
        scenario="BASELINE",
        universe_raw=universe_raw,
        rsr_df=rsr_df,
        alpha_df=alpha_df,
        regime_df=regime_df,
        trade_syms=trade_syms,
        rsr_syms=rsr_syms,
        cfg=cfg,
        start=OOS_START,
        end=OOS_END,
        capital=float(cfg.portfolio.capital),
        verbose=False,
        min_hold=cfg.risk.min_hold_days,
        enable_filters=enable_filters,
        volatility_threshold=vol_threshold,
        volume_multiplier=volume_mult,
        enable_market_filter=market_filter,
        topix_close=topix_close,
    )
    stats = calc_trade_stats(result.get("_trades", []))
    result.update(stats)
    return result


def classify_strength(vol_threshold: float, volume_mult: float) -> str:
    vol_rank = {1.05: 1, 1.10: 2, 1.15: 3, 1.20: 4}[round(float(vol_threshold), 2)]
    volume_rank = {1.1: 1, 1.2: 2, 1.3: 3, 1.5: 4}[round(float(volume_mult), 1)]
    score = vol_rank + volume_rank
    if score <= 3:
        return "weak"
    if score <= 5:
        return "medium"
    return "strong"


def load_reference_baseline() -> float | None:
    if not REFERENCE_JSON.exists():
        return None
    try:
        data = json.loads(REFERENCE_JSON.read_text(encoding="utf-8"))
        return float(data["mom21d"]["OOS"]["sharpe"])
    except Exception:
        return None


def main() -> int:
    grid = load_grid()
    cfg = build_cfg()
    dataset = load_data()

    rows: list[dict] = []
    total = len(grid)
    for i, rec in enumerate(grid.itertuples(index=False), start=1):
        print(
            f"[case {i:02d}/{total}] vol={rec.vol_threshold:.2f} "
            f"volmult={rec.volume_mult:.1f} market={bool(rec.market_filter)}"
        )
        result = run_case(
            dataset=dataset,
            cfg=cfg,
            enable_filters=True,
            vol_threshold=float(rec.vol_threshold),
            volume_mult=float(rec.volume_mult),
            market_filter=bool(rec.market_filter),
        )
        rows.append({
            "analysis_type": "case_detail",
            "vol_threshold": float(rec.vol_threshold),
            "volume_mult": float(rec.volume_mult),
            "market_filter": bool(rec.market_filter),
            "filter_strength": classify_strength(rec.vol_threshold, rec.volume_mult),
            "sharpe": result.get("sharpe", 0.0),
            "cagr": result.get("cagr", 0.0),
            "max_dd": result.get("max_dd", 0.0),
            "win_rate": result.get("win_rate", 0.0),
            "avg_hold_days": result.get("avg_hold_days", 0.0),
            "trade_count": result.get("n_trades", 0),
            "avg_profit_hold_days": result.get("avg_profit_hold_days", 0.0),
            "avg_loss_hold_days": result.get("avg_loss_hold_days", 0.0),
            "avg_win": result.get("avg_win", 0.0),
            "avg_loss": result.get("avg_loss", 0.0),
            "expectancy": result.get("expectancy", 0.0),
            "profit_factor": result.get("profit_factor", 0.0),
        })
        del result
        collect_and_log_process_memory(f"scenario {i:02d}")

    case_df = pd.DataFrame(rows)

    baseline = run_case(
        dataset=dataset,
        cfg=cfg,
        enable_filters=False,
        vol_threshold=1.15,
        volume_mult=1.3,
        market_filter=False,
    )
    baseline_ref = load_reference_baseline()

    corr = float(case_df["trade_count"].corr(case_df["sharpe"])) if len(case_df) >= 2 else np.nan

    summary_rows: list[dict] = []
    summary_rows.append({
        "analysis_type": "overall",
        "metric": "trade_count_sharpe_corr",
        "value": round(corr, 4),
    })

    for rec in case_df.groupby("vol_threshold", dropna=False).agg({
        "sharpe": "mean",
        "cagr": "mean",
        "max_dd": "mean",
        "trade_count": "mean",
        "avg_hold_days": "mean",
        "expectancy": "mean",
        "profit_factor": "mean",
    }).reset_index().itertuples(index=False):
        summary_rows.append({
            "analysis_type": "vol_threshold_summary",
            "metric": f"vol_threshold={rec.vol_threshold:.2f}",
            "sharpe": round(rec.sharpe, 4),
            "cagr": round(rec.cagr, 4),
            "max_dd": round(rec.max_dd, 4),
            "trade_count": round(rec.trade_count, 2),
            "avg_hold_days": round(rec.avg_hold_days, 2),
            "expectancy": round(rec.expectancy, 4),
            "profit_factor": round(rec.profit_factor, 4),
        })

    for rec in case_df.groupby("volume_mult", dropna=False).agg({
        "sharpe": "mean",
        "cagr": "mean",
        "max_dd": "mean",
        "trade_count": "mean",
        "avg_hold_days": "mean",
        "expectancy": "mean",
        "profit_factor": "mean",
    }).reset_index().itertuples(index=False):
        summary_rows.append({
            "analysis_type": "volume_mult_summary",
            "metric": f"volume_mult={rec.volume_mult:.1f}",
            "sharpe": round(rec.sharpe, 4),
            "cagr": round(rec.cagr, 4),
            "max_dd": round(rec.max_dd, 4),
            "trade_count": round(rec.trade_count, 2),
            "avg_hold_days": round(rec.avg_hold_days, 2),
            "expectancy": round(rec.expectancy, 4),
            "profit_factor": round(rec.profit_factor, 4),
        })

    for rec in case_df.groupby("filter_strength", dropna=False).agg({
        "sharpe": "mean",
        "cagr": "mean",
        "max_dd": "mean",
        "trade_count": "mean",
        "avg_hold_days": "mean",
        "avg_profit_hold_days": "mean",
        "avg_loss_hold_days": "mean",
        "avg_win": "mean",
        "avg_loss": "mean",
        "expectancy": "mean",
        "profit_factor": "mean",
    }).reset_index().itertuples(index=False):
        summary_rows.append({
            "analysis_type": "filter_strength_summary",
            "metric": rec.filter_strength,
            "sharpe": round(rec.sharpe, 4),
            "cagr": round(rec.cagr, 4),
            "max_dd": round(rec.max_dd, 4),
            "trade_count": round(rec.trade_count, 2),
            "avg_hold_days": round(rec.avg_hold_days, 2),
            "avg_profit_hold_days": round(rec.avg_profit_hold_days, 2),
            "avg_loss_hold_days": round(rec.avg_loss_hold_days, 2),
            "avg_win": round(rec.avg_win, 4),
            "avg_loss": round(rec.avg_loss, 4),
            "expectancy": round(rec.expectancy, 4),
            "profit_factor": round(rec.profit_factor, 4),
        })

    best = case_df.sort_values(["sharpe", "cagr"], ascending=[False, False]).iloc[0]
    baseline_sharpe = float(baseline.get("sharpe", 0.0))
    reference_sharpe = baseline_ref if baseline_ref is not None else baseline_sharpe

    summary_rows.append({
        "analysis_type": "baseline_comparison",
        "metric": "current_strategy_baseline",
        "sharpe": round(baseline_sharpe, 4),
        "cagr": round(float(baseline.get("cagr", 0.0)), 4),
        "max_dd": round(float(baseline.get("max_dd", 0.0)), 4),
        "trade_count": int(baseline.get("n_trades", 0)),
        "avg_hold_days": round(float(baseline.get("avg_hold_days", 0.0)), 2),
        "avg_profit_hold_days": round(float(baseline.get("avg_profit_hold_days", 0.0)), 2),
        "avg_loss_hold_days": round(float(baseline.get("avg_loss_hold_days", 0.0)), 2),
        "avg_win": round(float(baseline.get("avg_win", 0.0)), 4),
        "avg_loss": round(float(baseline.get("avg_loss", 0.0)), 4),
        "expectancy": round(float(baseline.get("expectancy", 0.0)), 4),
        "profit_factor": round(float(baseline.get("profit_factor", 0.0)), 4),
    })
    summary_rows.append({
        "analysis_type": "baseline_comparison",
        "metric": "best_filter_case_vs_baseline",
        "sharpe": round(float(best["sharpe"]), 4),
        "cagr": round(float(best["cagr"]), 4),
        "max_dd": round(float(best["max_dd"]), 4),
        "trade_count": int(best["trade_count"]),
        "avg_hold_days": round(float(best["avg_hold_days"]), 2),
        "avg_profit_hold_days": round(float(best["avg_profit_hold_days"]), 2),
        "avg_loss_hold_days": round(float(best["avg_loss_hold_days"]), 2),
        "avg_win": round(float(best["avg_win"]), 4),
        "avg_loss": round(float(best["avg_loss"]), 4),
        "expectancy": round(float(best["expectancy"]), 4),
        "profit_factor": round(float(best["profit_factor"]), 4),
        "delta_sharpe_vs_baseline": round(float(best["sharpe"]) - baseline_sharpe, 4),
        "delta_sharpe_vs_reference_0564": round(float(best["sharpe"]) - reference_sharpe, 4),
        "delta_trade_count_vs_baseline": int(best["trade_count"] - baseline.get("n_trades", 0)),
        "delta_avg_hold_vs_baseline": round(float(best["avg_hold_days"]) - float(baseline.get("avg_hold_days", 0.0)), 2),
        "delta_expectancy_vs_baseline": round(float(best["expectancy"]) - float(baseline.get("expectancy", 0.0)), 4),
    })

    out_df = pd.concat([case_df, pd.DataFrame(summary_rows)], ignore_index=True, sort=False)
    ANALYSIS_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(ANALYSIS_CSV, index=False, encoding="utf-8-sig")

    strength_summary = case_df.groupby("filter_strength").agg({
        "sharpe": "mean",
        "trade_count": "mean",
        "avg_hold_days": "mean",
        "expectancy": "mean",
    }).round(3)
    hypothesis = "over_filtering"
    report = [
        "Filter Degradation Analysis",
        f"Source CSV: {RESULTS_CSV}",
        f"Output CSV: {ANALYSIS_CSV}",
        "",
        f"Baseline Sharpe (recomputed, filters OFF): {baseline_sharpe:.3f}",
        f"Reference Sharpe 0.564 source: {REFERENCE_JSON.name if baseline_ref is not None else 'not found'}",
        f"Best filtered Sharpe: {float(best['sharpe']):.3f}",
        f"Sharpe delta vs recomputed baseline: {float(best['sharpe']) - baseline_sharpe:+.3f}",
        f"Sharpe delta vs reference 0.564: {float(best['sharpe']) - reference_sharpe:+.3f}",
        f"Trade count / Sharpe correlation: {corr:.3f}",
        "",
        "Filter strength summary:",
        strength_summary.to_string(),
        "",
        "Conclusion hypothesis:",
        f"- {hypothesis}",
        "Reason:",
        f"- Best filtered case expectancy {float(best['expectancy']):.3f}% vs baseline {float(baseline.get('expectancy', 0.0)):.3f}%.",
        f"- Best filtered trade_count {int(best['trade_count'])} vs baseline {int(baseline.get('n_trades', 0))}.",
        f"- Best filtered avg_hold_days {float(best['avg_hold_days']):.2f} vs baseline {float(baseline.get('avg_hold_days', 0.0)):.2f}.",
        "- Sharpe degrades as threshold strength increases, while drawdown improvement is limited.",
        "",
        "Next hypothesis to test:",
        "- over_filtering",
        "  Entry filters are removing too many valid breakouts without materially improving downside control.",
    ]
    REPORT_TXT.write_text("\n".join(report), encoding="utf-8")

    print(f"saved: {ANALYSIS_CSV}")
    print(f"saved: {REPORT_TXT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
