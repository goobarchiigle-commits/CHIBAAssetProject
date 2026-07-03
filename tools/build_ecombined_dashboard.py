"""
build_ecombined_dashboard.py
Study47 E_COMBINED Full IS → CSV export → dashboard PNG (v2 investor-grade)

Single command:
    python tools/build_ecombined_dashboard.py

Outputs:
    reports/dashboard_data/equity_curve.csv
    reports/dashboard_data/trades.csv
    reports/dashboard_data/exposure.csv
    reports/strategy_dashboard_ecombined_v2.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from src.backtest.snapshot_archaeology_202606 import build_common_dataset
import src.backtest.composite_alpha_bt as cab
from src.config_loader import load_strategy_config

# ── Study47 E_COMBINED parameters ─────────────────────────────────────────
IS_START  = "2018-01-01"
IS_END    = "2024-12-31"
DATA_END  = "2025-12-31"
CAPITAL   = 3_000_000
MIN_HOLD  = 3
VOL_CALM_THRESHOLD = 0.008
ADDON_SIZE_FRAC    = 0.25
ADDON_ATR_MULT     = 1.0

OUT_DIR   = ROOT / "reports" / "dashboard_data"
DASH_OUT  = ROOT / "reports" / "strategy_dashboard_ecombined_v2.png"


# ── helpers ────────────────────────────────────────────────────────────────
def build_vol_adj_ts(topix_close: pd.Series, dates) -> pd.Series:
    ret = topix_close.pct_change()
    std = ret.rolling(20, min_periods=10).std()
    std_s = std.reindex(pd.Index(dates), method="ffill").fillna(std.median())
    mpts  = pd.Series(3, index=std_s.index, dtype=int)
    mpts[std_s < VOL_CALM_THRESHOLD] = 4
    return mpts


def run_e_combined(ds: dict, sym_active_df, vol_adj_ts) -> dict:
    return cab.run_scenario(
        scenario               = "BASELINE",
        universe_raw           = ds["universe_raw"],
        rsr_df                 = ds["rsr_df"],
        alpha_df               = None,
        regime_df              = ds["regime_df"],
        trade_syms             = ds["trade_syms"],
        rsr_syms               = ds["rsr_syms"],
        cfg                    = ds["base_cfg"],
        start                  = IS_START,
        end                    = IS_END,
        verbose                = False,
        tech_matrices          = ds["tech_matrices"],
        breadth_series         = ds["breadth_series"],
        capital                = CAPITAL,
        min_hold               = MIN_HOLD,
        topix_close            = ds["topix_close"],
        market_shock_mode      = "composite",
        rsr_exit_threshold     = 70.0,
        sym_active_df          = sym_active_df,
        enable_simple_rsr_exit = True,
        enable_atr_trailing_prod = True,
        enable_multilayer_rsr  = True,
        enable_atr_risk_sizing = False,
        enable_mtf_filter      = False,
        sizing_mode            = "existing",
        exit_policy            = "A",
        exit_policy_atr_mult   = ADDON_ATR_MULT,
        exit_policy_defer_days = 5,
        max_positions_ts       = vol_adj_ts,
        addon_policy           = "D",
        addon_atr_mult         = ADDON_ATR_MULT,
        addon_stage2_mult      = 2.0,
        addon_max_per_pos      = 1,
        addon_size_frac        = ADDON_SIZE_FRAC,
    )


# ── CSV export ─────────────────────────────────────────────────────────────
def export_equity(result: dict, out: Path) -> None:
    eq: pd.Series = result["equity_curve"]
    df = eq.rename_axis("date").reset_index()
    df.columns = ["date", "equity"]
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df.to_csv(out, index=False)
    print(f"  equity_curve.csv  : {len(df)} rows  [{df['date'].iloc[0]} … {df['date'].iloc[-1]}]")


def export_trades(result: dict, out: Path) -> None:
    eq: pd.Series = result["equity_curve"]
    date_arr = np.array(eq.index)

    rows = []
    for t in result.get("_trades", []):
        if t.get("side") != "SELL" or t.get("pnl") is None:
            continue
        ei = int(t["entry_idx"])
        xi = int(t["exit_idx"])
        if ei >= len(date_arr) or xi >= len(date_arr):
            continue
        entry_px = float(t["entry"])
        exit_px  = float(t["exit"])
        ret_pct  = (exit_px - entry_px) / entry_px * 100.0 if entry_px > 0 else 0.0
        rows.append({
            "entry_date": str(date_arr[ei])[:10],
            "exit_date":  str(date_arr[xi])[:10],
            "symbol":     str(t["symbol"]),
            "ret_pct":    round(ret_pct, 4),
            "pnl_yen":    round(float(t["pnl"]), 0),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    win_rate = (df["ret_pct"] > 0).mean() * 100 if len(df) else 0
    print(f"  trades.csv        : {len(df)} rows  WinRate={win_rate:.1f}%")


def export_exposure(result: dict, out: Path) -> None:
    eq: pd.Series  = result["equity_curve"]
    ln: pd.Series  = result["long_notional"]
    date_arr       = np.array(eq.index)
    n              = len(date_arr)

    exposure_pct = (ln.values / np.maximum(eq.values, 1.0) * 100.0).round(2)

    pos_count = np.zeros(n, dtype=int)
    for t in result.get("_trades", []):
        if t.get("side") != "SELL":
            continue
        ei = int(t.get("entry_idx", -1))
        xi = int(t.get("exit_idx",  -1))
        if ei < 0 or xi < 0:
            continue
        lo = max(0, ei)
        hi = min(n, xi)
        if lo < hi:
            pos_count[lo:hi] += 1

    df = pd.DataFrame({
        "date":      [str(d)[:10] for d in date_arr],
        "exposure":  exposure_pct,
        "positions": pos_count,
    })
    df.to_csv(out, index=False)
    print(f"  exposure.csv      : {len(df)} rows  "
          f"AvgExp={exposure_pct.mean():.1f}%  AvgPos={pos_count.mean():.2f}")


# ── benchmark loading ──────────────────────────────────────────────────────
def load_benchmarks(ds: dict, equity: "pd.Series") -> dict:
    """Return dict of pd.Series aligned and normalized to equity.index[0]."""
    import yfinance as yf

    benchmarks = {}

    # TOPIX from project dataset
    topix_raw = ds["topix_close"]
    topix_aligned = topix_raw.reindex(equity.index, method="ffill").dropna()
    if len(topix_aligned) >= 2:
        benchmarks["TOPIX"] = topix_aligned
        print(f"  TOPIX  : n={len(topix_aligned)}  "
              f"start={topix_aligned.iloc[0]:.2f}  end={topix_aligned.iloc[-1]:.2f}")

    # Nikkei225 from yfinance
    try:
        raw = yf.download(
            "^N225",
            start=equity.index[0].strftime("%Y-%m-%d"),
            end=(equity.index[-1] + pd.Timedelta(days=3)).strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )
        if not raw.empty:
            close = raw["Close"]
            # flatten MultiIndex columns (ticker level)
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = close.squeeze()
            close.index = pd.to_datetime(close.index).tz_localize(None)
            nk_aligned = close.reindex(equity.index, method="ffill").dropna()
            if len(nk_aligned) >= 2:
                benchmarks["Nikkei225"] = nk_aligned
                v0 = float(nk_aligned.iloc[0])
                v1 = float(nk_aligned.iloc[-1])
                print(f"  Nikkei : n={len(nk_aligned)}  start={v0:.2f}  end={v1:.2f}")
    except Exception as e:
        print(f"  Nikkei225 fetch failed: {e}")

    return benchmarks


# ── benchmark metrics reporting ────────────────────────────────────────────
def _bench_cagr(series: "pd.Series") -> float:
    n = len(series) / 252
    return ((series.iloc[-1] / series.iloc[0]) ** (1.0 / n) - 1.0) * 100.0 if n > 0 else 0.0


def _bench_mult(series: "pd.Series") -> float:
    return float(series.iloc[-1] / series.iloc[0])


def report_benchmarks(benchmarks: dict, equity_cagr: float, equity_mult: float) -> None:
    print(f"\n{'─'*60}")
    print("  Benchmark Comparison  (aligned to equity data range)")
    print(f"{'─'*60}")
    print(f"  {'Name':>14}  {'CAGR':>7}  {'Final Multiple':>14}")
    print(f"  {'E_COMBINED':>14}  {equity_cagr:>6.2f}%  {equity_mult:>12.2f}x")
    for name, series in benchmarks.items():
        c = _bench_cagr(series)
        m = _bench_mult(series)
        print(f"  {name:>14}  {c:>6.2f}%  {m:>12.2f}x")
    print(f"{'─'*60}")


# ── metric verification ────────────────────────────────────────────────────
def verify(result: dict) -> None:
    TARGET = {"cagr": 20.51, "max_dd": -19.81, "sharpe": 0.880, "n_trades": 205}
    print(f"\n{'─'*60}")
    print("  Verification vs Study47 E_COMBINED")
    print(f"{'─'*60}")
    for k, target in TARGET.items():
        measured = result.get(k, float("nan"))
        if k == "n_trades":
            diff = int(measured) - int(target)
            ok = abs(diff) == 0
            print(f"  {k:<12}: measured={int(measured):>4}  target={int(target):>4}  "
                  f"Δ={diff:+d}  {'✓' if ok else '✗'}")
        else:
            diff = float(measured) - float(target)
            ok = abs(diff) < 0.05
            print(f"  {k:<12}: measured={float(measured):>7.3f}  "
                  f"target={float(target):>7.3f}  "
                  f"Δ={diff:+.3f}  {'✓' if ok else '✗'}")
    print(f"{'─'*60}")


# ── main ───────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("build_ecombined_dashboard.py  (v2 investor-grade)")
    print(f"  Period : {IS_START} – {IS_END}")
    print(f"  Capital: ¥{CAPITAL:,}")
    print("=" * 60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. dataset
    print("\n[1/5] データセット構築...")
    ds = build_common_dataset(DATA_END)
    all_syms  = list(ds["trade_syms"].keys())
    all_dates = sorted(set.union(*[
        set(ds["universe_raw"][s]["df"].index)
        for s in all_syms if s in ds["universe_raw"]
    ]))
    vol_adj_ts = build_vol_adj_ts(ds["topix_close"], all_dates)

    cfg  = load_strategy_config()
    bc   = cfg.risk_controls.bear_universe_filter
    be   = list(bc.excluded_sectors) if bc.enabled else None
    sym_active = cab.build_dyn_rsr42_active(
        universe_raw       = ds["universe_raw"],
        topix_close        = ds["topix_close"],
        rsr_df             = ds["rsr_df"],
        all_syms           = all_syms,
        start              = IS_START,
        end                = IS_END,
        bear_exclude_sectors = be,
        sym_sector_map     = dict(ds["trade_syms"]) if be else None,
    )

    # 2. run E_COMBINED
    print("\n[2/5] E_COMBINED バックテスト実行...")
    result = run_e_combined(ds, sym_active, vol_adj_ts)
    print(f"  CAGR={result['cagr']:+.2f}%  Sharpe={result['sharpe']:.3f}"
          f"  MaxDD={result['max_dd']:.2f}%  Trades={result['n_trades']}")

    # 3. CSV export
    print("\n[3/5] CSV エクスポート...")
    export_equity(result,   OUT_DIR / "equity_curve.csv")
    export_trades(result,   OUT_DIR / "trades.csv")
    export_exposure(result, OUT_DIR / "exposure.csv")

    # 4. load CSVs + benchmarks
    print("\n[4/5] ベンチマーク取得 + ダッシュボード生成...")
    from tools.generate_strategy_dashboard import (
        load_equity, load_trades, load_exposure, build_dashboard,
        bench_cagr, bench_final_mult,
    )
    equity   = load_equity(OUT_DIR / "equity_curve.csv")
    trades   = load_trades(OUT_DIR / "trades.csv")
    exposure = load_exposure(OUT_DIR / "exposure.csv")

    benchmarks = load_benchmarks(ds, equity)

    eq_start = equity.index[0].strftime("%Y-%m-%d")
    eq_end   = equity.index[-1].strftime("%Y-%m-%d")
    title_lines = [
        "CHIBAAssetProject E_COMBINED",
        f"Backtest Range: {IS_START} → {IS_END}",
        f"Equity Data Range: {eq_start} → {eq_end}",
    ]

    build_dashboard(equity, trades, DASH_OUT, title_lines,
                    exposure=exposure, benchmarks=benchmarks)

    # 5. verify + benchmark report
    verify(result)

    equity_cagr = result["cagr"]
    equity_mult = float(equity.iloc[-1] / equity.iloc[0])
    report_benchmarks(benchmarks, equity_cagr, equity_mult)

    print(f"\n出力:")
    print(f"  {OUT_DIR / 'equity_curve.csv'}")
    print(f"  {OUT_DIR / 'trades.csv'}")
    print(f"  {OUT_DIR / 'exposure.csv'}")
    print(f"  {DASH_OUT}")


if __name__ == "__main__":
    main()
