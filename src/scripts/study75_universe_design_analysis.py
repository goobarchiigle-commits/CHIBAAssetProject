"""
src/scripts/study75_universe_design_analysis.py
Study75 Universe Design — descriptive analysis only (no backtest, no parameter optimization).
Reads already-downloaded data/jquants/processed/ + metadata/universe_events.parquet.
Writes backtests/study75_universe_design_2026-07-10.json (raw numbers cited by
reports/study75_universe_design.md).

Design principle applied throughout: all liquidity/lot-cost filter evaluations use POINT-IN-TIME
snapshots (trailing 20 trading days as of a given date), never a single "current" static value
applied retroactively — a static filter would look ahead and introduce a NEW survivorship-style
bias (GUARD_CRITICAL: lookahead=forbid).
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json

import numpy as np
import pandas as pd

from src.paths import JQUANTS_METADATA_DIR, JQUANTS_PROCESSED_DIR, RESULTS_DIR

LOT_SIZE = 100  # matches src/backtest/composite_alpha_bt.py LOT constant
CAPITAL_LEVELS = [3_000_000, 10_000_000, 20_000_000, 30_000_000]  # matches Study42/43A/74 capital grid
MAX_LOT_COST_RATIO = 0.30  # strategy.yaml max_lot_cost_ratio
ADV20_THRESHOLD_JPY = 30_000_000  # illustrative liquidity floor, not backtest-tuned
TOPIX500_PROXY_SIZE = 500


def main() -> int:
    print("[1/6] Loading processed daily_bars_{year}.parquet (Code, Date, Close, Volume only)...")
    year_files = sorted(JQUANTS_PROCESSED_DIR.glob("daily_bars_*.parquet"))
    frames = [pd.read_parquet(f, columns=["Date", "Code", "Close", "Volume"]) for f in year_files]
    panel = pd.concat(frames, ignore_index=True)
    panel["Date"] = pd.to_datetime(panel["Date"])
    panel["Year"] = panel["Date"].dt.year
    print(f"  rows={len(panel):,} symbols={panel['Code'].nunique():,} years={sorted(panel['Year'].unique())}")

    symbols_per_year = panel.groupby("Year")["Code"].nunique().to_dict()

    print("[2/6] Loading universe_events.parquet for delisted/year...")
    events = pd.read_parquet(JQUANTS_METADATA_DIR / "universe_events.parquet")
    events["event_date"] = pd.to_datetime(events["event_date"])
    events["Year"] = events["event_date"].dt.year
    delisted_per_year = events.loc[events["event_type"] == "REMOVE"].groupby("Year")["code"].nunique().to_dict()
    added_per_year = events.loc[events["event_type"] == "ADD"].groupby("Year")["code"].nunique().to_dict()

    print("[3/6] Computing traded value distribution (raw Close x raw Volume)...")
    panel["TradedValue"] = panel["Close"] * panel["Volume"]
    sym_year_liquidity = panel.groupby(["Year", "Code"])["TradedValue"].mean().reset_index()
    liquidity_percentiles = {
        int(year): {f"p{p}": float(np.percentile(grp["TradedValue"].dropna(), p)) for p in (10, 25, 50, 75, 90, 99)}
        for year, grp in sym_year_liquidity.groupby("Year")
    }

    print("[4/6] Computing lot-cost distribution...")
    sym_year_price = panel.groupby(["Year", "Code"])["Close"].mean().reset_index()
    sym_year_price["LotCost"] = sym_year_price["Close"] * LOT_SIZE
    lot_cost_percentiles = {}
    lot_infeasible_ratio = {}
    for year, grp in sym_year_price.groupby("Year"):
        lot_cost_percentiles[int(year)] = {
            f"p{p}": float(np.percentile(grp["LotCost"].dropna(), p)) for p in (10, 25, 50, 75, 90, 99)
        }
        row = {}
        for cap in CAPITAL_LEVELS:
            threshold = cap * MAX_LOT_COST_RATIO
            infeasible = (grp["LotCost"] > threshold).sum()
            row[str(cap)] = {"infeasible_count": int(infeasible), "infeasible_pct": round(100 * infeasible / len(grp), 1)}
        lot_infeasible_ratio[int(year)] = row

    print("[5/6] Computing point-in-time filter snapshots (ADV20 as of each year-end)...")
    panel_sorted = panel.sort_values(["Code", "Date"])
    filter_snapshots = {}
    for year in sorted(panel["Year"].unique()):
        year_end = panel.loc[panel["Year"] == year, "Date"].max()
        window_start = year_end - pd.Timedelta(days=40)  # generous window to ensure >=20 trading days
        window = panel_sorted.loc[(panel_sorted["Date"] > window_start) & (panel_sorted["Date"] <= year_end)]
        adv20 = window.groupby("Code").agg(
            adv20_value=("TradedValue", "mean"),
            last_close=("Close", "last"),
            n_days=("Date", "count"),
        )
        adv20 = adv20.loc[adv20["n_days"] >= 15]
        adv20["lot_cost"] = adv20["last_close"] * LOT_SIZE

        b_pass = adv20.loc[adv20["adv20_value"] >= ADV20_THRESHOLD_JPY]
        c_pass = b_pass.loc[b_pass["lot_cost"] <= CAPITAL_LEVELS[0] * MAX_LOT_COST_RATIO]
        d_pass = adv20.sort_values("adv20_value", ascending=False).head(TOPIX500_PROXY_SIZE)

        filter_snapshots[int(year)] = {
            "A_all": len(adv20),
            "B_adv20": len(b_pass),
            "C_adv20_lot": len(c_pass),
            "D_topix500_proxy": len(d_pass),
        }

    print("[6/6] Writing raw results JSON...")
    out = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "lot_size": LOT_SIZE,
        "capital_levels": CAPITAL_LEVELS,
        "max_lot_cost_ratio": MAX_LOT_COST_RATIO,
        "adv20_threshold_jpy": ADV20_THRESHOLD_JPY,
        "topix500_proxy_size": TOPIX500_PROXY_SIZE,
        "symbols_per_year": {int(k): int(v) for k, v in symbols_per_year.items()},
        "delisted_per_year": {int(k): int(v) for k, v in delisted_per_year.items()},
        "added_per_year": {int(k): int(v) for k, v in added_per_year.items()},
        "liquidity_percentiles_jpy": liquidity_percentiles,
        "lot_cost_percentiles_jpy": lot_cost_percentiles,
        "lot_infeasible_by_capital": lot_infeasible_ratio,
        "filter_snapshots_year_end": filter_snapshots,
    }
    out_path = RESULTS_DIR / "study75_universe_design_2026-07-10.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
