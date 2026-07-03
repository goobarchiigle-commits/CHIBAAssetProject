from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

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


OOS_START = "2025-01-01"
OOS_END = "2025-12-31"
FULL_START = "2018-01-01"
FULL_END = OOS_END

VOL_THRESHOLDS = [1.05, 1.10, 1.15, 1.20]
VOLUME_MULTS = [1.1, 1.2, 1.3, 1.5]
MARKET_FILTERS = [True, False]

OUTPUT_CSV = REPO_ROOT / "results" / "filter_sensitivity_oos.csv"


def load_data():
    print("[1/3] loading universe and price data...")
    rsr_syms = _bt._load_rsr_universe()
    trade_syms = rsr_syms
    all_syms = {**rsr_syms, **trade_syms}
    universe_raw = download_universe(all_syms, start=FULL_START, end=FULL_END, verbose=False)
    topix_close = _bt._download_topix(FULL_START, FULL_END)

    print("[2/3] computing rsr and alpha...")
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

    print("[3/3] computing regime frame...")
    regime_df = _bt._calc_regime(topix_close)
    return universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms, topix_close


def run_one(
    universe_raw,
    rsr_df,
    alpha_df,
    regime_df,
    trade_syms,
    rsr_syms,
    topix_close,
    cfg,
    vol_threshold: float,
    volume_mult: float,
    market_filter: bool,
) -> dict:
    return _bt.run_scenario(
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
        enable_filters=True,
        volatility_threshold=vol_threshold,
        volume_multiplier=volume_mult,
        enable_market_filter=market_filter,
        topix_close=topix_close,
    )


def main() -> int:
    cfg = load_strategy_config()
    cfg = replace(
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

    print("=" * 80)
    print("  filter sensitivity test / OOS 2025")
    print("  fixed params: mom_period=21 turtle_exit=55 min_hold=3 min_rsr=75 max_pos=3 capital=3,000,000")
    print("=" * 80)

    dataset = load_data()

    rows: list[dict] = []
    total = len(VOL_THRESHOLDS) * len(VOLUME_MULTS) * len(MARKET_FILTERS)
    count = 0
    for vol_threshold in VOL_THRESHOLDS:
        for volume_mult in VOLUME_MULTS:
            for market_filter in MARKET_FILTERS:
                count += 1
                print(
                    f"[{count:02d}/{total}] vol_threshold={vol_threshold:.2f} "
                    f"volume_mult={volume_mult:.1f} market_filter={market_filter}"
                )
                result = run_one(*dataset, cfg, vol_threshold, volume_mult, market_filter)
                rows.append({
                    "vol_threshold": vol_threshold,
                    "volume_mult": volume_mult,
                    "market_filter": market_filter,
                    "sharpe": result.get("sharpe", 0.0),
                    "cagr": result.get("cagr", 0.0),
                    "max_dd": result.get("max_dd", 0.0),
                    "win_rate": result.get("win_rate", 0.0),
                    "avg_hold_days": result.get("avg_hold_days", 0.0),
                    "trade_count": result.get("n_trades", 0),
                })
                del result
                collect_and_log_process_memory(f"scenario {count:02d}")

    df = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\nTop 5 by Sharpe")
    print(df.sort_values(["sharpe", "cagr"], ascending=[False, False]).head(5).to_string(index=False))
    print(f"\nSaved: {OUTPUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
