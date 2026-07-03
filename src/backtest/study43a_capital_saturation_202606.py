"""
backtest/study43a_capital_saturation_202606.py

Study43A — Capital Saturation Risk Audit
目的: 現行戦略（APRIL_REPRO_A config）にとって最適な運用資本を特定する。
     最大CAGRではなくリスク調整後パフォーマンスと資本効率の最大化が目標。

Fixed: lot_size=100, max_positions=3, topix_close=None, sizing_mode="existing"
Capital sweep: 3M / 5M / 10M / 15M / 20M / 30M / 50M / 100M

Output:
  backtests/study43a_capital_saturation_202606_<date>.json
  backtests/study43a_capital_saturation_202606_<date>_charts.png
"""
from __future__ import annotations

import sys
import json
import warnings
from pathlib import Path
from datetime import date as _date

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams["font.family"] = "MS Gothic"
plt.rcParams["axes.unicode_minus"] = False

from src.backtest import composite_alpha_bt as cab
from src.backtest.snapshot_archaeology_202606 import build_common_dataset
from src.paths import RESULTS_DIR

# ── 定数 ────────────────────────────────────────────────────────────────
IS_START = "2018-01-01"
IS_END   = "2024-12-31"
LOT      = 100

CAPITAL_LEVELS = [
    3_000_000, 5_000_000, 10_000_000, 15_000_000,
    20_000_000, 30_000_000, 50_000_000, 100_000_000,
]
LABELS = ["¥3M", "¥5M", "¥10M", "¥15M", "¥20M", "¥30M", "¥50M", "¥100M"]

# Study42 REPRO_A確定値（参照）
REPRO_A_CAGR   = 20.14
REPRO_A_SHARPE = 0.859
REPRO_A_DD     = -15.71


def run_repro_a(ds: dict, capital: int) -> dict:
    """APRIL_REPRO_A config + 指定資本で run_scenario を実行。"""
    return cab.run_scenario(
        scenario             = "BASELINE",
        universe_raw         = ds["universe_raw"],
        rsr_df               = ds["rsr_df"],
        alpha_df             = None,
        regime_df            = ds["regime_df"],
        trade_syms           = ds["trade_syms"],
        rsr_syms             = ds["rsr_syms"],
        cfg                  = ds["base_cfg"],
        start                = IS_START,
        end                  = IS_END,
        verbose              = False,
        tech_matrices        = ds["tech_matrices"],
        breadth_series       = ds["breadth_series"],
        capital              = capital,
        min_hold             = 3,
        market_shock_mode    = "full_exit",
        rsr_exit_threshold   = 75.0,
        sym_active_df        = None,
        enable_simple_rsr_exit       = True,
        enable_atr_trailing_prod     = False,
        enable_multilayer_rsr        = False,
        enable_atr_risk_sizing       = False,
        enable_mtf_filter            = False,
        use_fixed_pct_trail          = False,
        sizing_mode                  = "existing",
        lot_size             = LOT,
    )


def compute_monthly_worst_quarter(monthly_returns: dict) -> float:
    """月次リターン辞書から最悪四半期リターン（連続3ヶ月合算）を計算。"""
    if not monthly_returns:
        return float("nan")
    months = sorted(monthly_returns.keys())
    if len(months) < 3:
        return min(monthly_returns.values())
    worst = float("inf")
    for i in range(len(months) - 2):
        q = (monthly_returns[months[i]] + monthly_returns[months[i + 1]]
             + monthly_returns[months[i + 2]])
        if q < worst:
            worst = q
    return round(worst, 2)


def compute_peak_trough_period(equity_curve: pd.Series) -> dict:
    """equity_curve から最大DDのピーク→トラフ期間と回復日数を計算。"""
    if equity_curve is None or len(equity_curve) < 2:
        return {"peak_date": None, "trough_date": None,
                "peak_to_trough_days": None, "recovery_days": None,
                "total_underwater_days": None}

    rolling_max = equity_curve.cummax()
    dd_series = (equity_curve - rolling_max) / rolling_max

    # 最大DD地点
    trough_idx = dd_series.idxmin()
    trough_val = dd_series[trough_idx]

    # ピーク（トラフ以前の直近高値）
    peak_idx = equity_curve[:trough_idx].idxmax()

    # 回復: トラフ以降で rolling_max を超えた最初の日
    post_trough = equity_curve[trough_idx:]
    peak_val = float(equity_curve[peak_idx])
    recovered = post_trough[post_trough >= peak_val]
    recovery_idx = recovered.index[0] if not recovered.empty else None

    peak_date   = str(peak_idx.date()) if hasattr(peak_idx, "date") else str(peak_idx)
    trough_date = str(trough_idx.date()) if hasattr(trough_idx, "date") else str(trough_idx)

    # カレンダー日数
    try:
        pt_days = (trough_idx - peak_idx).days
    except Exception:
        pt_days = None

    if recovery_idx is not None:
        try:
            rec_days = (recovery_idx - trough_idx).days
        except Exception:
            rec_days = None
        rec_date = str(recovery_idx.date()) if hasattr(recovery_idx, "date") else str(recovery_idx)
    else:
        rec_days = None
        rec_date = "未回復"

    # 水中日数（DD < 0の日数）
    underwater = int((dd_series < -0.001).sum())

    return {
        "peak_date":           peak_date,
        "trough_date":         trough_date,
        "recovery_date":       rec_date,
        "peak_to_trough_days": pt_days,
        "recovery_days":       rec_days,
        "total_underwater_days": underwater,
    }


def compute_lot_fwd20d(rejected_detail: list[dict],
                        universe_raw: dict,
                        all_common_dates: list) -> dict:
    """Lot拒否された候補の fwd_20d 平均リターンを計算。"""
    date_idx = {
        str(d.date()) if hasattr(d, "date") else str(d): i
        for i, d in enumerate(all_common_dates)
    }
    fwd_list: list[float] = []
    for rec in rejected_detail:
        sym  = rec.get("symbol", "")
        dstr = rec.get("date", "")
        if sym not in universe_raw or dstr not in date_idx:
            continue
        idx = date_idx[dstr]
        if idx >= len(all_common_dates):
            continue
        close_s = universe_raw[sym]["df"]["Close"]
        entry_date = all_common_dates[idx]
        if entry_date not in close_s.index:
            continue
        epx = float(close_s[entry_date])
        if epx <= 0:
            continue
        fwd_dates = all_common_dates[idx + 1: idx + 21]
        fwd_cls = close_s.reindex(fwd_dates).dropna()
        if fwd_cls.empty:
            continue
        fwd_list.append(float(fwd_cls.iloc[-1]) / epx - 1.0)
    if not fwd_list:
        return {"avg_fwd20d_pct": None, "n": 0, "pct_positive": None}
    return {
        "avg_fwd20d_pct": round(np.mean(fwd_list) * 100, 2),
        "n": len(fwd_list),
        "pct_positive": round(sum(1 for r in fwd_list if r > 0) / len(fwd_list) * 100, 1),
    }


def extract_full(res: dict, universe_raw: dict, all_common_dates: list) -> dict:
    """全診断指標を抽出。"""
    ec: pd.Series | None = res.get("equity_curve")
    monthly: dict = res.get("monthly_returns", {}) or {}
    dd_summary: dict = res.get("drawdown_summary", {}) or {}

    # 基本性能
    cagr   = float(res.get("cagr",   0.0) or 0.0)
    sharpe = float(res.get("sharpe", 0.0) or 0.0)
    maxdd  = float(res.get("max_dd", 0.0) or 0.0)
    calmar = float(res.get("calmar", 0.0) or 0.0)

    # 月次/四半期
    worst_month = round(min(monthly.values()), 2) if monthly else None
    worst_month_date = (min(monthly, key=lambda k: monthly[k])
                        if monthly else None)
    worst_quarter = compute_monthly_worst_quarter(monthly)

    # Peak-to-trough & recovery
    pt = compute_peak_trough_period(ec)

    # Capacity diagnostics
    lot_rej    = int(res.get("rejected_by_lot_count", 0) or 0)
    missed_cap = int(res.get("missed_by_cap_count", 0) or 0)
    idle_pct   = res.get("avg_idle_cash_ratio_pct")
    sat_pct    = res.get("cap_saturation_rate_pct")
    avg_pos    = res.get("avg_simultaneous_holdings")
    days_max   = res.get("days_at_max_positions")

    # Lot fwd 20d
    lot_detail = res.get("_rejected_by_lot_detail", []) or []
    lot_fwd20  = compute_lot_fwd20d(lot_detail, universe_raw, all_common_dates)

    # Missed entry fwd 20d (cap-missed)
    missed_cands = (res.get("_missed_cands", []) or [])[:2000]
    missed_fwd20 = compute_lot_fwd20d(missed_cands, universe_raw, all_common_dates)

    return {
        # Performance
        "cagr":   round(cagr,   3),
        "sharpe": round(sharpe, 3),
        "max_dd": round(maxdd,  3),
        "calmar": round(calmar, 3),
        "n_trades": int(res.get("n_trades", 0) or 0),
        "sortino":  round(float(res.get("sortino", 0.0) or 0.0), 3),
        "profit_factor": round(float(res.get("profit_factor", 0.0) or 0.0), 3),
        "win_rate": round(float(res.get("win_rate", 0.0) or 0.0), 1),
        "annual_returns": monthly and {
            yr: round(float(res.get("annual_returns", {}).get(yr, 0.0)), 1)
            for yr in (res.get("annual_returns", {}) or {})
        } or {},
        "annual_returns_raw": res.get("annual_returns", {}),
        # Risk diagnostics
        "worst_month_pct":  worst_month,
        "worst_month_date": worst_month_date,
        "worst_quarter_pct": worst_quarter,
        "peak_date":           pt["peak_date"],
        "trough_date":         pt["trough_date"],
        "recovery_date":       pt.get("recovery_date"),
        "peak_to_trough_days": pt["peak_to_trough_days"],
        "recovery_days":       pt["recovery_days"],
        "total_underwater_days": pt["total_underwater_days"],
        "time_underwater_pct": dd_summary.get("time_underwater_pct"),
        # Capacity diagnostics
        "avg_simultaneous_holdings": round(float(avg_pos or 0.0), 2),
        "days_at_max_positions":    int(days_max or 0),
        "cap_saturation_rate_pct":  sat_pct,
        "avg_idle_cash_pct":        idle_pct,
        "lot_reject_count":         lot_rej,
        "missed_by_cap_count":      missed_cap,
        "missed_fwd20d":            missed_fwd20,
        "lot_fwd20d":               lot_fwd20,
    }


def make_charts(rows: list[dict], labels: list[str], out_path: Path) -> None:
    """6チャート生成して PNG 保存。"""
    caps_m = [r["capital"] / 1e6 for r in rows]
    cagrs   = [r["cagr"]    for r in rows]
    sharpes = [r["sharpe"]  for r in rows]
    maxdds  = [abs(r["max_dd"]) for r in rows]
    calmars = [r["calmar"]  for r in rows]
    idles   = [r.get("avg_idle_cash_pct") or 0 for r in rows]
    lotrejs = [r["lot_reject_count"] for r in rows]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Study43A — Capital Saturation Risk Audit\n"
                 f"APRIL_REPRO_A config  IS={IS_START}..{IS_END}",
                 fontsize=14, fontweight="bold")

    def _plot(ax, y, title, ylabel, color="steelblue", marker="o"):
        ax.plot(caps_m, y, color=color, marker=marker, linewidth=2, markersize=7)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Capital (M¥)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"¥{int(x)}M"))
        ax.grid(True, alpha=0.35)
        ax.tick_params(axis="x", rotation=30)
        for xi, yi, lb in zip(caps_m, y, labels):
            ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=7.5)

    _plot(axes[0, 0], cagrs,   "Capital vs CAGR",    "CAGR (%)",       "steelblue")
    _plot(axes[0, 1], sharpes, "Capital vs Sharpe",  "Sharpe",          "darkorange")
    _plot(axes[0, 2], maxdds,  "Capital vs MaxDD",   "|MaxDD| (%)",     "crimson")
    _plot(axes[1, 0], calmars, "Capital vs Calmar",  "Calmar",          "seagreen")
    _plot(axes[1, 1], idles,   "Capital vs Idle Cash","Idle Cash (%)",  "purple")
    _plot(axes[1, 2], lotrejs, "Capital vs Lot Reject","Lot Reject Count","saddlebrown")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved: {out_path}")


def main() -> int:
    print("=" * 78)
    print("  Study43A — Capital Saturation Risk Audit")
    print(f"  IS={IS_START}..{IS_END}  lot={LOT}  max_pos=3  sizing=existing")
    print(f"  REPRO_A baseline: CAGR={REPRO_A_CAGR}%  Sh={REPRO_A_SHARPE}  DD={REPRO_A_DD}%")
    print("=" * 78)

    # ── 1. データセット ──────────────────────────────────────────────────
    print(f"\n[1] データセット構築（end={IS_END}）...")
    ds = build_common_dataset(IS_END)
    all_syms = list(ds["trade_syms"].keys())
    print(f"  完了 ({len(all_syms)} syms)")

    date_sets = [set(ds["universe_raw"][s]["df"].index)
                 for s in all_syms if s in ds["universe_raw"]]
    all_common_dates: list = sorted(set.intersection(*date_sets)) if date_sets else []

    # ── 2. Capital Sweep ─────────────────────────────────────────────────
    print(f"\n[2] Capital Sweep ({len(CAPITAL_LEVELS)} levels)...")
    sweep_rows: list[dict] = []

    for cap, lbl in zip(CAPITAL_LEVELS, LABELS):
        print(f"\n  ── {lbl} (capital={cap:,}) ──")
        res_raw = run_repro_a(ds, capital=cap)
        m = extract_full(res_raw, ds["universe_raw"], all_common_dates)
        m["capital"] = cap
        m["label"]   = lbl
        sweep_rows.append(m)
        ann = res_raw.get("annual_returns", {}) or {}
        print(f"    CAGR={m['cagr']:+.2f}%  Sh={m['sharpe']:.3f}  "
              f"DD={m['max_dd']:.2f}%  Calmar={m['calmar']:.3f}")
        print(f"    Trades={m['n_trades']}  LotRej={m['lot_reject_count']}"
              f"  MissedCap={m['missed_by_cap_count']}"
              f"  Idle={m['avg_idle_cash_pct']}%"
              f"  Sat={m['cap_saturation_rate_pct']}%")
        print(f"    WorstMo={m['worst_month_pct']}%({m['worst_month_date']})"
              f"  WorstQ={m['worst_quarter_pct']}%"
              f"  PT_days={m['peak_to_trough_days']}"
              f"  RecovDays={m['recovery_days']}")
        if ann:
            print(f"    Annual: " + "  ".join(
                f"{yr}={v:+.1f}%" for yr, v in sorted(ann.items())))

    # ── 3. Marginal Efficiency ───────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  [3] Marginal Capital Efficiency")
    print("=" * 78)
    print(f"\n  {'Label':<8}  {'Capital':>10}  {'CAGR':>7}  {'ΔShp':>7}  "
          f"{'ΔCalmar':>8}  {'Sh':>6}  {'Calmar':>7}  {'LotRej':>7}")
    print("  " + "-" * 72)
    prev = None
    marginals: list[dict] = []
    for row in sweep_rows:
        if prev is None:
            dcagr  = 0.0
            dsharpe = 0.0
            dcalmar = 0.0
        else:
            dcagr   = round(row["cagr"]   - prev["cagr"],   3)
            dsharpe = round(row["sharpe"] - prev["sharpe"], 3)
            dcalmar = round(row["calmar"] - prev["calmar"], 3)
        marginals.append({"label": row["label"], "delta_cagr": dcagr,
                           "delta_sharpe": dsharpe, "delta_calmar": dcalmar})
        print(f"  {row['label']:<8}  {row['capital']:>10,}  "
              f"{row['cagr']:>+7.2f}%  {dsharpe:>+7.3f}  {dcalmar:>+8.3f}  "
              f"{row['sharpe']:>6.3f}  {row['calmar']:>7.3f}  "
              f"{row['lot_reject_count']:>7}")
        prev = row

    # ── 4. Analysis ──────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  [4] Analysis")
    print("=" * 78)

    cagrs   = [r["cagr"]   for r in sweep_rows]
    sharpes = [r["sharpe"] for r in sweep_rows]
    calmars = [r["calmar"] for r in sweep_rows]

    # A. Saturation point (ΔCAGR < 0.5pp)
    sat_point = None
    for i in range(1, len(sweep_rows)):
        if cagrs[i] - cagrs[i - 1] < 0.5:
            sat_point = sweep_rows[i]["label"]
            break

    # B. Max Calmar capital
    best_calmar_idx = int(np.argmax(calmars))
    best_calmar_lbl = sweep_rows[best_calmar_idx]["label"]
    best_calmar_val = calmars[best_calmar_idx]

    # C. Best risk-adjusted (Sharpe)
    best_sharpe_idx = int(np.argmax(sharpes))
    best_sharpe_lbl = sweep_rows[best_sharpe_idx]["label"]
    best_sharpe_val = sharpes[best_sharpe_idx]

    # D. Primary driver (lot vs capital)
    # lot_reject=0 first appears at...
    lot_cliff_lbl = None
    for row in sweep_rows:
        if row["lot_reject_count"] == 0:
            lot_cliff_lbl = row["label"]
            break

    r_3m  = next(r for r in sweep_rows if r["capital"] == 3_000_000)
    r_lot0 = next((r for r in sweep_rows if r["lot_reject_count"] == 0), sweep_rows[-1])
    cagr_at_lot0 = r_lot0["cagr"]
    cagr_gain_to_lot0 = cagr_at_lot0 - r_3m["cagr"]

    r_30m  = next((r for r in sweep_rows if r["capital"] == 30_000_000), None)
    r_100m = next((r for r in sweep_rows if r["capital"] == 100_000_000), None)
    cagr_beyond_lot0 = (r_100m["cagr"] - r_30m["cagr"]) if (r_30m and r_100m) else 0.0

    # E. Practical ranges
    # Minimum Efficient: first point where lot_reject=0
    min_efficient = lot_cliff_lbl
    # Optimal: max Calmar
    optimal = best_calmar_lbl
    # Saturation: where ΔCAGR < 0.5pp
    saturation = sat_point

    print(f"\n  A. Saturation point (ΔCAGR < 0.5pp): {sat_point or 'not reached'}")
    print(f"  B. Max Calmar: {best_calmar_lbl}  Calmar={best_calmar_val:.3f}")
    print(f"  C. Best Sharpe: {best_sharpe_lbl}  Sharpe={best_sharpe_val:.3f}")
    print(f"  D. Primary driver analysis:")
    print(f"     Lot cliff (reject=0 first at): {lot_cliff_lbl}")
    print(f"     CAGR gain to lot cliff: +{cagr_gain_to_lot0:.2f}pp")
    print(f"     CAGR gain beyond lot cliff (30M→100M): +{cagr_beyond_lot0:.2f}pp")
    if (cagr_gain_to_lot0 + cagr_beyond_lot0) > 0.01:
        pct_lot = cagr_gain_to_lot0 / (cagr_gain_to_lot0 + cagr_beyond_lot0) * 100
        print(f"     Lot removal = {pct_lot:.0f}% of total gain;  "
              f"Pure capital expansion beyond = {100-pct_lot:.0f}%")
    print(f"  E. Practical operating ranges:")
    print(f"     Minimum Efficient Capital: {min_efficient} (lot_reject=0 first)")
    print(f"     Optimal Capital (max Calmar): {optimal}")
    print(f"     Saturation Capital: {saturation or '¥30M以上'}")

    # ── 5. ¥30M Risk Profile vs ¥3M ─────────────────────────────────────
    print("\n" + "=" * 78)
    print("  [5] ¥30M vs ¥3M Risk Profile Comparison")
    print("=" * 78)
    r_3m_row  = next(r for r in sweep_rows if r["capital"] == 3_000_000)
    r_30m_row = next((r for r in sweep_rows if r["capital"] == 30_000_000), None)
    if r_30m_row:
        fields = [
            ("CAGR (%)",               "cagr",              "+.2f"),
            ("Sharpe",                 "sharpe",            ".3f"),
            ("MaxDD (%)",              "max_dd",            ".2f"),
            ("Calmar",                 "calmar",            ".3f"),
            ("Worst Month (%)",        "worst_month_pct",   ".2f"),
            ("Worst Quarter (%)",      "worst_quarter_pct", ".2f"),
            ("Peak-to-Trough days",    "peak_to_trough_days", "d"),
            ("Recovery days",          "recovery_days",      ""),
            ("Underwater days",        "total_underwater_days", "d"),
            ("Idle Cash (%)",          "avg_idle_cash_pct", ""),
            ("Lot Reject",             "lot_reject_count",  "d"),
            ("Missed by Cap",          "missed_by_cap_count","d"),
        ]
        print(f"\n  {'Metric':<26}  {'¥3M':>10}  {'¥30M':>10}  {'Delta':>10}")
        print("  " + "-" * 60)
        for label, key, fmt in fields:
            v3  = r_3m_row.get(key)
            v30 = r_30m_row.get(key)
            if v3 is None or v30 is None:
                print(f"  {label:<26}  {str(v3):>10}  {str(v30):>10}  {'N/A':>10}")
                continue
            try:
                dv = v30 - v3 if isinstance(v3, (int, float)) else "N/A"
                fstr = f"{v3:{fmt}}" if fmt not in ("", "d") else str(v3)
                fstr30 = f"{v30:{fmt}}" if fmt not in ("", "d") else str(v30)
                fdelta = f"{dv:{'+.2f' if isinstance(dv, float) else ''}}" if isinstance(dv, (int, float)) else "N/A"
                print(f"  {label:<26}  {fstr:>10}  {fstr30:>10}  {fdelta:>10}")
            except Exception:
                print(f"  {label:<26}  {str(v3):>10}  {str(v30):>10}")
        # MaxDD in yen terms
        print(f"\n  MaxDD in ¥ terms:")
        print(f"    ¥3M:  ¥{3_000_000 * abs(r_3m_row['max_dd']) / 100:>10,.0f}"
              f"  ({r_3m_row['max_dd']:.2f}%)")
        print(f"    ¥30M: ¥{30_000_000 * abs(r_30m_row['max_dd']) / 100:>10,.0f}"
              f"  ({r_30m_row['max_dd']:.2f}%)")

    # ── 6. Final Recommendation ──────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  [6] Final Recommendation")
    print("=" * 78)

    # Find recommended range
    # Lower bound: min_efficient (lot_reject=0)
    # Upper bound: saturation or max Calmar
    rec_low  = lot_cliff_lbl
    rec_high = best_calmar_lbl
    rec_row  = sweep_rows[best_calmar_idx]

    print(f"\n  Recommended Capital Range: {rec_low} – {rec_high}")
    print(f"\n  Expected Metrics @ {rec_high}:")
    print(f"    CAGR:   {rec_row['cagr']:+.2f}%")
    print(f"    Sharpe: {rec_row['sharpe']:.3f}")
    print(f"    MaxDD:  {rec_row['max_dd']:.2f}%")
    print(f"    Calmar: {rec_row['calmar']:.3f}")
    print(f"    LotRej: {rec_row['lot_reject_count']}")

    if r_30m_row:
        print(f"\n  If scaled to ¥30M:")
        print(f"    Expected CAGR:   {r_30m_row['cagr']:+.2f}%  "
              f"(vs ¥3M {r_3m_row['cagr']:+.2f}%, Δ={r_30m_row['cagr']-r_3m_row['cagr']:+.2f}pp)")
        print(f"    Expected MaxDD:  {r_30m_row['max_dd']:.2f}%  "
              f"(in ¥: ¥{30_000_000 * abs(r_30m_row['max_dd']) / 100:,.0f})")
        print(f"    Risk profile shift: DD% {r_3m_row['max_dd']:.2f}% → {r_30m_row['max_dd']:.2f}%  "
              f"({r_30m_row['max_dd']-r_3m_row['max_dd']:+.2f}pp)")
        print(f"    Calmar: {r_30m_row['calmar']:.3f}  "
              f"(vs ¥3M {r_3m_row['calmar']:.3f}, Δ={r_30m_row['calmar']-r_3m_row['calmar']:+.3f})")
        print(f"    Worst Month: {r_30m_row['worst_month_pct']}%  "
              f"(vs ¥3M {r_3m_row['worst_month_pct']}%)")

    # ── 7. Charts ───────────────────────────────────────────────────────
    print("\n[7] チャート生成...")
    today_str = str(_date.today())
    out_dir = Path(RESULTS_DIR) / ".." / "backtests"
    out_dir.mkdir(parents=True, exist_ok=True)
    chart_path = out_dir / f"study43a_capital_saturation_202606_{today_str}_charts.png"
    make_charts(sweep_rows, LABELS, chart_path)

    # ── 8. JSON 保存 ────────────────────────────────────────────────────
    out = {
        "study": "Study43A",
        "date": today_str,
        "period": {"is_start": IS_START, "is_end": IS_END},
        "config": {"lot_size": LOT, "max_positions": 3,
                   "sizing_mode": "existing", "topix_close": None},
        "sweep": [
            {k: v for k, v in r.items()
             if k not in ("annual_returns",)}
            for r in sweep_rows
        ],
        "marginals": marginals,
        "analysis": {
            "saturation_point":          sat_point,
            "max_calmar_capital":        best_calmar_lbl,
            "max_calmar_value":          round(best_calmar_val, 3),
            "best_sharpe_capital":       best_sharpe_lbl,
            "best_sharpe_value":         round(best_sharpe_val, 3),
            "lot_cliff_capital":         lot_cliff_lbl,
            "cagr_gain_to_lot_cliff_pp": round(cagr_gain_to_lot0, 3),
            "cagr_gain_beyond_lot_cliff_pp": round(cagr_beyond_lot0, 3),
            "min_efficient_capital":     min_efficient,
            "optimal_capital":           optimal,
            "saturation_capital":        saturation,
        },
        "recommendation": {
            "range_low":    rec_low,
            "range_high":   rec_high,
            "expected_cagr":   rec_row["cagr"],
            "expected_sharpe": rec_row["sharpe"],
            "expected_max_dd": rec_row["max_dd"],
            "expected_calmar": rec_row["calmar"],
        },
        "vs_3m_comparison": {
            k: {
                "3m":  r_3m_row.get(k),
                "30m": r_30m_row.get(k) if r_30m_row else None,
            }
            for k in ("cagr", "sharpe", "max_dd", "calmar",
                      "worst_month_pct", "worst_quarter_pct",
                      "peak_to_trough_days", "recovery_days",
                      "lot_reject_count", "missed_by_cap_count")
        },
    }
    json_path = out_dir / f"study43a_capital_saturation_202606_{today_str}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f"  JSON saved: {json_path}")

    print("\n" + "=" * 78)
    print(f"  Study43A 完了")
    print(f"  Min Efficient: {min_efficient}  Optimal(Calmar): {optimal}"
          f"  Saturation: {saturation or '¥30M+'}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
