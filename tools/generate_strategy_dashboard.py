"""
generate_strategy_dashboard.py  (v2)
Strategy performance dashboard generator.

Layout:
  [A: Equity + Benchmarks (full-width)]
  [B: Drawdown | C: Calendar-Year Returns]
  [D: Summary Metrics | E: Trade Distribution]
  [F: Exposure & Positions (full-width)]  ← optional

Usage:
    python tools/generate_strategy_dashboard.py \\
        --equity  equity_curve.csv \\
        --trades  trades.csv \\
        --exposure exposure.csv \\
        --out     reports/out.png

equity_curve.csv  : date, equity
trades.csv        : exit_date (or date/entry_date), ret_pct (or ret/ret_adj/return)
exposure.csv      : date, exposure [, positions]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

matplotlib.rcParams["font.family"] = "MS Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

TRADING_DAYS = 252

FIG_BG    = "white"
AX_BG     = "white"
GRID_CLR  = "#dddddd"
TEXT_CLR  = "#222222"
TICK_CLR  = "#555555"
SPINE_CLR = "#cccccc"

C_EQUITY  = "#2e8b57"
C_DD_LINE = "#2e8b57"
C_DD_FILL = "#a8d5b5"
C_BAR_CAL = "#4472c4"
C_BAR_NEG = "#c0392b"
C_CAGR    = "#3a9e5f"
C_MAXDD   = "#b04040"
C_SHARPE  = "#7b5ea7"
C_CALMAR  = "#5b8db8"
C_DIST    = "#4d9e58"
C_EXP     = "#2e8b57"
C_POS     = "#8b4513"
C_BENCH   = {"TOPIX": "#e07b2a", "Nikkei225": "#7b7bc8"}


# ── loaders ────────────────────────────────────────────────────────────────
def load_equity(path: Path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["date"])
    if "equity" not in df.columns:
        raise ValueError(f"'equity' column not found. Got: {list(df.columns)}")
    return df.sort_values("date").set_index("date")["equity"].astype(float)


def load_trades(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_col = next((c for c in ("exit_date", "date", "entry_date") if c in df.columns), None)
    if date_col is None:
        raise ValueError("trades.csv must have exit_date / date / entry_date")
    df["_date"] = pd.to_datetime(df[date_col])

    ret_col = next((c for c in ("ret_pct", "ret", "ret_adj", "return") if c in df.columns), None)
    if ret_col is None:
        raise ValueError("trades.csv must have ret_pct / ret / ret_adj / return")
    raw = df[ret_col].astype(float)
    df["_ret"] = raw if abs(raw.median()) > 0.5 else raw * 100.0

    df["_pnl"]  = df["pnl_yen"].astype(float) if "pnl_yen" in df.columns \
                  else (df["pnl"].astype(float) if "pnl" in df.columns else np.nan)
    df["_year"] = df["_date"].dt.year
    return df


def load_exposure(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").set_index("date")
    if "exposure" not in df.columns:
        return None
    df["exposure"] = df["exposure"].astype(float)
    if "positions" in df.columns:
        df["positions"] = df["positions"].astype(float)
    return df


# ── metrics ────────────────────────────────────────────────────────────────
def _dd_series(equity: pd.Series) -> pd.Series:
    return (equity - equity.cummax()) / equity.cummax() * 100.0


def _cagr(equity: pd.Series) -> float:
    n = len(equity) / TRADING_DAYS
    return ((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / n) - 1.0) * 100.0 if n > 0 else 0.0


def _max_dd(equity: pd.Series) -> float:
    return float(_dd_series(equity).min())


def _max_dd_date(equity: pd.Series) -> pd.Timestamp:
    return _dd_series(equity).idxmin()


def _sharpe(equity: pd.Series) -> float:
    r = equity.pct_change().dropna()
    return float(r.mean() / r.std() * TRADING_DAYS**0.5) if r.std() > 0 else 0.0


def _calmar(equity: pd.Series) -> float:
    c = _cagr(equity) / 100.0
    d = abs(_max_dd(equity) / 100.0)
    return round(c / d, 3) if d > 0 else 0.0


def _pf(trades: pd.DataFrame) -> float:
    r  = trades["_ret"]
    gw = r[r > 0].sum()
    gl = abs(r[r < 0].sum())
    return float(gw / gl) if gl > 0 else float("inf")


def _yearly_returns(equity: pd.Series) -> pd.Series:
    last  = equity.resample("YE").last()
    first = equity.resample("YE").first()
    ret   = (last / first - 1.0) * 100.0
    ret.index = ret.index.year
    return ret


def _yearly_maxdd(equity: pd.Series) -> pd.Series:
    dd = _dd_series(equity)
    result = {yr: float(grp.min()) for yr, grp in dd.groupby(dd.index.year)}
    return pd.Series(result)


def _scalar(v) -> float:
    """Safely coerce a possibly-Series scalar to float."""
    if hasattr(v, "item"):
        return float(v.item())
    if hasattr(v, "iloc"):
        return float(v.iloc[0])
    return float(v)


def _align_bench(bench: pd.Series, equity: pd.Series) -> pd.Series:
    s = bench.squeeze() if hasattr(bench, "squeeze") else bench
    return s.reindex(equity.index, method="ffill").dropna()


def bench_cagr(bench: pd.Series, equity: pd.Series) -> float:
    """CAGR of benchmark aligned to equity date range."""
    aligned = _align_bench(bench, equity)
    if len(aligned) < 2:
        return float("nan")
    n  = len(aligned) / TRADING_DAYS
    v0 = _scalar(aligned.iloc[0])
    v1 = _scalar(aligned.iloc[-1])
    return ((v1 / v0) ** (1.0 / n) - 1.0) * 100.0 if n > 0 and v0 > 0 else 0.0


def bench_final_mult(bench: pd.Series, equity: pd.Series) -> float:
    aligned = _align_bench(bench, equity)
    if len(aligned) < 2:
        return float("nan")
    v0 = _scalar(aligned.iloc[0])
    v1 = _scalar(aligned.iloc[-1])
    return v1 / v0 if v0 > 0 else float("nan")


def compute_metrics(equity, trades, exposure_df) -> dict:
    avg_exp = float(exposure_df["exposure"].mean())  if exposure_df is not None else 0.0
    med_exp = float(exposure_df["exposure"].median()) if exposure_df is not None else 0.0
    max_pos = int(exposure_df["positions"].max())     \
              if (exposure_df is not None and "positions" in exposure_df.columns) else 0
    return {
        "final_mult": equity.iloc[-1] / equity.iloc[0],
        "CAGR":    _cagr(equity),
        "MaxDD":   _max_dd(equity),
        "MaxDDDt": _max_dd_date(equity),
        "Sharpe":  _sharpe(equity),
        "Calmar":  _calmar(equity),
        "PF":      _pf(trades),
        "Trades":  len(trades),
        "WinRate": float((trades["_ret"] > 0).mean() * 100.0),
        "AvgTrade":float(trades["_ret"].mean()),
        "AvgExp":  avg_exp,
        "MedExp":  med_exp,
        "MaxPos":  max_pos,
    }


# ── axis style ─────────────────────────────────────────────────────────────
def _style(ax, *, xlabel="", ylabel="", title="") -> None:
    ax.set_facecolor(AX_BG)
    ax.tick_params(colors=TICK_CLR, labelsize=8)
    for lab in (ax.xaxis.label, ax.yaxis.label, ax.title):
        lab.set_color(TEXT_CLR)
    for sp in ax.spines.values():
        sp.set_edgecolor(SPINE_CLR)
    ax.grid(color=GRID_CLR, linewidth=0.6, alpha=0.8)
    if xlabel: ax.set_xlabel(xlabel, fontsize=9, color=TEXT_CLR)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9, color=TEXT_CLR)
    if title:  ax.set_title(title,  fontsize=10, color=TEXT_CLR, pad=6)


# ── Panel A — Equity + Benchmarks ─────────────────────────────────────────
def panel_equity(ax, equity, metrics, title_lines, benchmarks: dict) -> None:
    initial  = float(equity.iloc[0])
    multiple = equity / initial

    # strategy line
    ax.plot(equity.index, multiple.values, color=C_EQUITY,
            linewidth=1.2, zorder=4, label="E_COMBINED")
    ax.fill_between(equity.index, multiple.values, 1.0,
                    alpha=0.10, color=C_EQUITY)

    # benchmark lines
    for name, bseries in benchmarks.items():
        aligned = bseries.reindex(equity.index, method="ffill").dropna()
        bmult   = aligned / aligned.iloc[0]
        clr     = C_BENCH.get(name, "#999999")
        ax.plot(aligned.index, bmult.values, color=clr,
                linewidth=0.9, linestyle="--", zorder=3, alpha=0.85, label=name)

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=[1, 2, 5]))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:.1f}x" if v < 2 else f"{v:.0f}x")
    )
    _style(ax, ylabel="Equity multiple, log scale")

    # multi-line title (left-aligned, top of panel)
    ax.set_title("\n".join(title_lines), fontsize=9, color=TEXT_CLR,
                 loc="left", pad=8, fontweight="bold", linespacing=1.5)

    # stats box — strategy metrics + benchmark comparison table
    m = metrics
    lines = [
        f"Final {m['final_mult']:.2f}x  CAGR {m['CAGR']:.1f}%  "
        f"DD {m['MaxDD']:.1f}%  Sharpe {m['Sharpe']:.2f}  Calmar {m['Calmar']:.2f}",
        f"Trades {int(m['Trades'])}  Win {m['WinRate']:.1f}%  "
        f"AvgTrade {m['AvgTrade']:+.1f}%",
    ]
    if benchmarks:
        lines.append("─" * 46)
        lines.append(f"{'':>12}  {'CAGR':>6}  {'Final':>6}")
        lines.append(f"{'E_COMBINED':>12}  {m['CAGR']:>5.1f}%  "
                     f"{float(m['final_mult']):>5.2f}x")
        for bname, bseries in benchmarks.items():
            bc = bench_cagr(bseries, equity)
            bm = bench_final_mult(bseries, equity)
            lines.append(f"{bname:>12}  {bc:>5.1f}%  {bm:>5.2f}x")

    ax.text(0.01, 0.97, "\n".join(lines), transform=ax.transAxes,
            va="top", ha="left", fontsize=8, color=TEXT_CLR,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.40", facecolor="white",
                      edgecolor=SPINE_CLR, alpha=0.90))

    # legend
    if benchmarks:
        leg = ax.legend(fontsize=8, loc="lower right",
                        facecolor="white", edgecolor=SPINE_CLR, framealpha=0.85)
        for t in leg.get_texts():
            t.set_color(TEXT_CLR)


# ── Panel B — Drawdown (fixed Y-axis + MaxDD annotation) ──────────────────
def panel_drawdown(ax, equity) -> None:
    dd = _dd_series(equity)
    ax.fill_between(dd.index, dd.values, 0,
                    where=dd.values <= 0, alpha=0.50, color=C_DD_FILL, zorder=2)
    ax.plot(dd.index, dd.values, color=C_DD_LINE, linewidth=0.8, zorder=3)
    ax.axhline(0, color=SPINE_CLR, linewidth=0.8)

    # fixed Y-axis
    ax.set_ylim(-22, 2)
    ax.set_yticks([0, -5, -10, -15, -20])
    ax.set_yticklabels(["0%", "-5%", "-10%", "-15%", "-20%"], fontsize=8)

    # MaxDD annotation
    mdd_val  = float(dd.min())
    mdd_date = dd.idxmin()
    ax.annotate(
        f"MaxDD\n{mdd_val:.2f}%\n{mdd_date.strftime('%Y-%m-%d')}",
        xy=(mdd_date, mdd_val),
        xytext=(mdd_date, mdd_val - 0.5),
        textcoords="data",
        ha="center", va="top",
        fontsize=7, color=C_MAXDD,
        arrowprops=dict(arrowstyle="-", color=C_MAXDD, lw=0.8),
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                  edgecolor=C_MAXDD, alpha=0.80),
    )
    _style(ax, title="Drawdown")


# ── Panel C — Calendar-Year Returns + per-year MaxDD ──────────────────────
def panel_yearly(ax, equity) -> None:
    yr   = _yearly_returns(equity)
    yrdd = _yearly_maxdd(equity)

    x = np.arange(len(yr))
    w = 0.38

    # Return bars (left of each group)
    ret_colors = [C_BAR_CAL if v >= 0 else C_BAR_NEG for v in yr.values]
    ax.bar(x - w / 2, yr.values, width=w, color=ret_colors, zorder=3)

    # MaxDD bars (right of each group, always negative, muted red)
    ax.bar(x + w / 2, yrdd.values, width=w, color="#d9a0a0", zorder=3)

    ax.axhline(0, color=SPINE_CLR, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(yr.index.astype(str), fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    # compact table annotation: Year | Ret | MaxDD
    hdr  = f"{'Year':>4}  {'Ret':>6}  {'MaxDD':>6}"
    rows = [hdr, "─" * 22]
    for yr_i in yr.index:
        r = yr.get(yr_i, float("nan"))
        d = yrdd.get(yr_i, float("nan"))
        sign = "+" if r >= 0 else ""
        rows.append(f"{yr_i:>4}  {sign}{r:>5.1f}%  {d:>6.1f}%")
    ax.text(0.99, 0.98, "\n".join(rows), transform=ax.transAxes,
            va="top", ha="right", fontsize=7, color=TEXT_CLR,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                      edgecolor=SPINE_CLR, alpha=0.88))

    # legend (manual patches)
    import matplotlib.patches as mpatches
    ax.legend(
        handles=[
            mpatches.Patch(color=C_BAR_CAL, label="Return"),
            mpatches.Patch(color="#d9a0a0", label="MaxDD"),
        ],
        fontsize=7, loc="upper left",
        facecolor="white", edgecolor=SPINE_CLR, framealpha=0.85,
    )
    _style(ax, ylabel="%", title="Calendar-Year Return & Max DD")


# ── Panel D — Summary Metrics + comparison table ──────────────────────────
def panel_metrics(ax, metrics, benchmarks, equity) -> None:
    labels = ["CAGR",       "Max DD",   "Sharpe",   "Calmar"]
    values = [metrics["CAGR"], metrics["MaxDD"], metrics["Sharpe"], metrics["Calmar"]]
    colors = [C_CAGR,       C_MAXDD,    C_SHARPE,   C_CALMAR]

    x    = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, width=0.55, zorder=3)
    ax.axhline(0, color=SPINE_CLR, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, color=TEXT_CLR)

    for bar, val in zip(bars, values):
        off = 0.3 if val >= 0 else -0.6
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + off,
                f"{val:.2f}",
                ha="center", va="bottom" if val >= 0 else "top",
                fontsize=7.5, color=TEXT_CLR)

    _style(ax, title="Summary Metrics")


# ── Panel E — Trade Return Distribution ───────────────────────────────────
def panel_distribution(ax, trades) -> None:
    rets   = trades["_ret"].dropna()
    n_bins = min(50, max(15, len(rets) // 4))
    ax.hist(rets, bins=n_bins, color=C_DIST, alpha=0.80,
            edgecolor="white", linewidth=0.3, zorder=3)
    ax.axvline(0, color=SPINE_CLR, linewidth=0.8)
    _style(ax, xlabel="Trade return (%)", ylabel="Count",
           title="Trade Return Distribution")


# ── Panel F — Exposure / Position Count ───────────────────────────────────
def panel_exposure(ax, exposure_df, metrics) -> None:
    ax2 = ax.twinx()
    exp = exposure_df["exposure"]

    ax.fill_between(exp.index, exp.values, 0, alpha=0.25, color=C_EXP, zorder=2)
    ax.plot(exp.index, exp.values, color=C_EXP, linewidth=0.9, zorder=3)
    ax.set_ylabel("Exposure (%)", fontsize=9, color=C_EXP)
    ax.tick_params(axis="y", colors=C_EXP, labelsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_ylim(0, max(100, float(exp.max()) * 1.1))

    if "positions" in exposure_df.columns:
        pos = exposure_df["positions"]
        ax2.step(pos.index, pos.values, color=C_POS, linewidth=0.8,
                 alpha=0.75, where="mid", zorder=4)
        ax2.set_ylabel("Positions", fontsize=9, color=C_POS)
        ax2.tick_params(axis="y", colors=C_POS, labelsize=8)
        ax2.set_ylim(0, max(6, int(pos.max()) + 1))
        ax2.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    else:
        ax2.set_yticks([])

    # stats annotation
    ann = (f"Avg Exp: {metrics['AvgExp']:.1f}%  |  "
           f"Median Exp: {metrics['MedExp']:.1f}%  |  "
           f"Max Positions: {metrics['MaxPos']}")
    ax.text(0.01, 0.96, ann, transform=ax.transAxes,
            va="top", ha="left", fontsize=8, color=TEXT_CLR,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=SPINE_CLR, alpha=0.85))

    _style(ax, title="Exposure & Position Count")
    ax.set_facecolor(AX_BG)
    for sp in ax.spines.values():
        sp.set_edgecolor(SPINE_CLR)
    ax.grid(color=GRID_CLR, linewidth=0.6, alpha=0.8)


# ── main build ─────────────────────────────────────────────────────────────
def build_dashboard(
    equity:      pd.Series,
    trades:      pd.DataFrame,
    out_path:    Path,
    title_lines: list[str],
    exposure:    Optional[pd.DataFrame] = None,
    benchmarks:  Optional[dict]         = None,
) -> None:
    if benchmarks is None:
        benchmarks = {}
    metrics  = compute_metrics(equity, trades, exposure)
    has_exp  = exposure is not None
    n_rows   = 4 if has_exp else 3
    h_ratios = [2.8, 1.5, 1.5, 1.2] if has_exp else [2.8, 1.5, 1.5]

    fig = plt.figure(figsize=(12, 14 if has_exp else 11), facecolor=FIG_BG)
    gs  = GridSpec(n_rows, 2, figure=fig,
                   hspace=0.50, wspace=0.32,
                   top=0.94, bottom=0.06, left=0.07, right=0.97,
                   height_ratios=h_ratios, width_ratios=[1, 1])

    ax_eq   = fig.add_subplot(gs[0, :])
    ax_dd   = fig.add_subplot(gs[1, 0])
    ax_cal  = fig.add_subplot(gs[1, 1])
    ax_met  = fig.add_subplot(gs[2, 0])
    ax_dist = fig.add_subplot(gs[2, 1])

    panel_equity(ax_eq,   equity, metrics, title_lines, benchmarks)
    panel_drawdown(ax_dd, equity)
    panel_yearly(ax_cal,  equity)
    panel_metrics(ax_met, metrics, benchmarks, equity)
    panel_distribution(ax_dist, trades)

    if has_exp:
        ax_exp = fig.add_subplot(gs[3, :])
        panel_exposure(ax_exp, exposure, metrics)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=FIG_BG)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--equity",   type=Path, default=Path("equity_curve.csv"))
    p.add_argument("--trades",   type=Path, default=Path("trades.csv"))
    p.add_argument("--exposure", type=Path, default=Path("exposure.csv"))
    p.add_argument("--out",      type=Path, default=Path("reports/strategy_dashboard.png"))
    p.add_argument("--title",    type=str,  default="")
    return p.parse_args()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = _parse_args()
    for p in (args.equity, args.trades):
        if not p.exists():
            print(f"ERROR: not found: {p}", file=sys.stderr); sys.exit(1)
    equity   = load_equity(args.equity)
    trades   = load_trades(args.trades)
    exposure = load_exposure(args.exposure)
    start = equity.index[0].strftime("%Y")
    end   = equity.index[-1].strftime("%Y")
    title_lines = [args.title or f"Strategy Dashboard: {start}-{end}"]
    build_dashboard(equity, trades, args.out, title_lines, exposure=exposure)


if __name__ == "__main__":
    main()
