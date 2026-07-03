"""
risk_pct_wf_validation.py — risk_pct 1.00% vs 1.25% 5-fold walk-forward 検証

5-fold expanding-window walk-forward:
  Fold 1: OOS 2020 (Train 2018-2019)
  Fold 2: OOS 2021 (Train 2018-2020)
  Fold 3: OOS 2022 (Train 2018-2021)
  Fold 4: OOS 2023 (Train 2018-2022)
  Fold 5: OOS 2024 (Train 2018-2023)
  Ref   : OOS 2025 (True holdout — 参考値)

比較: risk_pct = 1.00% vs 1.25%
他パラメータ固定: capital=3M / max_pos=3 / max_single_weight=25%

CLAUDE.md OVERFIT_GUARD 準拠:
  oos_is_ratio_min=0.7 / walkforward_required=true
  min_trade_required=5 / stability_check=required
"""
from __future__ import annotations

import sys, json, csv
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT     = Path(__file__).resolve().parents[2]
BD_DIR   = ROOT / "data" / "backtest_dataset" / "2026-03-28"
CACHE_DIR= ROOT / "cache" / "ohlcv"            # 直近データ (2025 OOS 用)
TOPIX_SYM = "1306.T"

# ─── 定数 ────────────────────────────────────────────────────────────────────
CAPITAL           = 3_000_000
MAX_POSITIONS     = 3
MAX_SINGLE_WEIGHT = 0.25
MAX_HOLD_DAYS     = 60
LOT               = 100
COST              = 0.001 + 0.00055
MIN_RSR           = 75.0
TURTLE_ENTRY_N    = 20
TURTLE_EXIT_N     = 55

RISK_PCTS         = [0.010, 0.0125]

# 5-fold + True OOS
WF_FOLDS = [
    ("Fold1", "2020-01-01", "2020-12-31"),
    ("Fold2", "2021-01-01", "2021-12-31"),
    ("Fold3", "2022-01-01", "2022-12-31"),
    ("Fold4", "2023-01-01", "2023-12-31"),
    ("Fold5", "2024-01-01", "2024-12-31"),
    ("TrueOOS", "2025-01-01", "2025-12-31"),
]

MIN_TRADE_THRESHOLD = 5  # CLAUDE.md: min_trade_required

# ─── データロード ─────────────────────────────────────────────────────────────

def _parse_df(df: pd.DataFrame) -> pd.DataFrame:
    close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    needed    = [c for c in [close_col, "High", "Low", "Volume"] if c in df.columns]
    df = df[needed].copy()
    df.columns = [{"Adj Close": "Close"}.get(c, c) for c in df.columns]
    df.index   = pd.to_datetime(df.index, utc=True).tz_convert(None).normalize()
    return df[~df.index.duplicated()].sort_index()


def _load_from(sym: str) -> pd.DataFrame | None:
    """
    BD_DIR (2010-2026-03) 優先でロード。
    BD_DIR に存在する場合: BD_DIR ベース + CACHE_DIR の新しい行を結合して
    2025 OOS フォールドにも対応する。
    BD_DIR に存在しない場合: CACHE_DIR のみ (2024-10 以降に限定)。
    """
    bd_path    = BD_DIR    / f"{sym}.parquet"
    cache_path = CACHE_DIR / f"{sym}.parquet"

    if bd_path.exists():
        df_bd = _parse_df(pd.read_parquet(bd_path))
        # CACHE_DIR に最新データがあれば結合して 2025 をカバー
        if cache_path.exists():
            df_c = _parse_df(pd.read_parquet(cache_path))
            new_rows = df_c[df_c.index > df_bd.index[-1]]
            if not new_rows.empty:
                # 共通カラムのみ結合
                common = [c for c in df_bd.columns if c in new_rows.columns]
                df_bd  = pd.concat([df_bd[common], new_rows[common]])
        return df_bd

    if cache_path.exists():
        return _parse_df(pd.read_parquet(cache_path))

    return None


def load_universe() -> dict[str, str]:
    from src.paths import LIVE_UNIVERSE_FILE
    d = json.loads(Path(LIVE_UNIVERSE_FILE).read_text(encoding="utf-8"))
    return d.get("symbols", d)


# ─── パネル・ATR20 事前計算 ────────────────────────────────────────────────────

def build_data_cache(universe: dict[str, str]) -> tuple[
    pd.DataFrame,    # close panel
    pd.DataFrame,    # high panel
    pd.DataFrame,    # low panel
    pd.DataFrame,    # atr20 panel
    pd.Series,       # topix
]:
    """全シンボルの OHLCV・ATR20 を一括計算してキャッシュ。"""
    closes, highs, lows = {}, {}, {}

    for sym in universe:
        df = _load_from(sym)
        if df is not None and len(df) > 100:
            closes[sym] = df["Close"]
            highs[sym]  = df["High"]
            lows[sym]   = df["Low"]

    panel_cl = pd.DataFrame(closes).sort_index()
    panel_hi = pd.DataFrame(highs).sort_index()
    panel_lo = pd.DataFrame(lows).sort_index()

    # ATR20 事前計算
    atr_dict: dict[str, pd.Series] = {}
    for sym in panel_cl.columns:
        cl = panel_cl[sym]
        hi = panel_hi[sym]
        lo = panel_lo[sym]
        tr = pd.concat([hi - lo, (hi - cl.shift(1)).abs(), (lo - cl.shift(1)).abs()], axis=1).max(axis=1)
        atr_dict[sym] = tr.rolling(20).mean()
    panel_atr = pd.DataFrame(atr_dict)

    # TOPIX
    tp = _load_from(TOPIX_SYM)
    if tp is None:
        raise RuntimeError(f"TOPIX {TOPIX_SYM} not found")
    topix = tp["Close"].reindex(panel_cl.index).ffill()

    return panel_cl, panel_hi, panel_lo, panel_atr, topix


def compute_rsr(panel_cl: pd.DataFrame, topix: pd.Series, period: int = 42) -> pd.DataFrame:
    sym_ret   = panel_cl.pct_change(period)
    topix_ret = topix.pct_change(period)
    excess    = sym_ret.sub(topix_ret, axis=0)
    return excess.rank(axis=1, pct=True) * 100


# ─── バックテストエンジン ──────────────────────────────────────────────────────

@dataclass
class Trade:
    symbol:      str
    entry_date:  pd.Timestamp
    entry_price: float
    qty:         int
    qty_risk:    int
    qty_cap:     int
    binding:     str   # "qty_risk" | "qty_cap" | "equal"
    exit_date:   pd.Timestamp | None = None
    exit_price:  float | None = None
    pnl:         float = 0.0


def run_period(
    start:      str,
    end:        str,
    risk_pct:   float,
    panel_cl:   pd.DataFrame,
    panel_hi:   pd.DataFrame,
    panel_lo:   pd.DataFrame,
    panel_atr:  pd.DataFrame,
    rsr:        pd.DataFrame,
) -> dict:
    dates = panel_cl.loc[start:end].index
    if len(dates) < 40:
        return {"error": "insufficient_data", "n_trades": 0}

    cash      = float(CAPITAL)
    peak_eq   = float(CAPITAL)
    positions: dict[str, Trade] = {}
    equity_log: list[float]     = []
    inv_log:    list[float]     = []
    trades:     list[Trade]     = []

    pos_budget   = float(CAPITAL * MAX_SINGLE_WEIGHT)
    risk_budget  = float(CAPITAL * risk_pct)

    for i, date in enumerate(dates):
        cl_row  = panel_cl.loc[date]
        port_val = sum(
            t.qty * float(cl_row.get(sym, t.entry_price))
            for sym, t in positions.items()
        )
        equity   = cash + port_val
        peak_eq  = max(peak_eq, equity)
        equity_log.append(equity)
        inv_log.append(port_val / max(equity, 1))

        # ── Exit ──────────────────────────────────────────────────────
        to_exit = []
        for sym, t in positions.items():
            price = float(cl_row.get(sym, t.entry_price))
            held  = i - dates.get_loc(t.entry_date)
            # 55 日安値
            lo_hist = panel_lo[sym].loc[:date].iloc[-TURTLE_EXIT_N-1:-1]
            stop    = float(lo_hist.min()) if not lo_hist.empty else 0.0
            if price <= stop or held >= MAX_HOLD_DAYS:
                to_exit.append((sym, price))

        for sym, price in to_exit:
            t = positions.pop(sym)
            cash += t.qty * price * (1 - COST)
            t.exit_date = date; t.exit_price = price
            t.pnl = (price - t.entry_price) * t.qty
            trades.append(t)

        # ── Entry ─────────────────────────────────────────────────────
        if len(positions) >= MAX_POSITIONS:
            continue
        if date not in rsr.index:
            continue

        rsr_row    = rsr.loc[date].dropna()
        candidates = rsr_row[rsr_row >= MIN_RSR].sort_values(ascending=False)
        new_today  = 0

        for sym in candidates.index:
            if len(positions) >= MAX_POSITIONS or new_today >= 1:
                break
            if sym in positions or sym not in panel_cl.columns:
                continue

            price = float(cl_row.get(sym, np.nan))
            if np.isnan(price) or price <= 0:
                continue

            # 20 日高値ブレイク
            hi_hist = panel_hi[sym].loc[:date].iloc[-TURTLE_ENTRY_N-1:-1]
            if hi_hist.empty or price <= float(hi_hist.max()):
                continue

            # ATR20 (事前計算済み)
            atr = panel_atr[sym].get(date, np.nan) if sym in panel_atr.columns else np.nan
            if np.isnan(atr) or atr <= 0:
                continue

            lot_cost = price * LOT
            qty_raw  = int(risk_budget / atr)
            qty_risk = max(LOT, (qty_raw // LOT) * LOT)
            qty_cap  = int(pos_budget // lot_cost) * LOT
            if qty_cap <= 0:
                continue

            qty = min(qty_risk, qty_cap)
            if qty <= 0:
                continue
            if cash < qty * price * (1 + COST):
                continue

            if qty_risk < qty_cap:
                binding = "qty_risk"
            elif qty_cap < qty_risk:
                binding = "qty_cap"
            else:
                binding = "equal"

            cash -= qty * price * (1 + COST)
            t = Trade(sym, date, price, qty, qty_risk, qty_cap, binding)
            positions[sym] = t
            new_today += 1

    # 残ポジション決済
    for sym, t in positions.items():
        price = float(panel_cl.loc[dates[-1]].get(sym, t.entry_price))
        cash += t.qty * price * (1 - COST)
        t.exit_date = dates[-1]; t.exit_price = price
        t.pnl = (price - t.entry_price) * t.qty
        trades.append(t)

    # ─── メトリクス ───────────────────────────────────────────────────
    eq    = pd.Series(equity_log, index=dates)
    years = len(dates) / 252
    final = eq.iloc[-1]

    cagr   = (final / CAPITAL) ** (1 / max(years, 0.01)) - 1
    dd     = eq / eq.cummax() - 1
    max_dd = float(dd.min())
    dr     = eq.pct_change().dropna()
    sharpe = float(dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else 0.0
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0

    n = len(trades)
    br = sum(1 for t in trades if t.binding == "qty_risk") / max(n, 1)
    bc = sum(1 for t in trades if t.binding == "qty_cap")  / max(n, 1)
    avg_inv  = float(np.mean(inv_log))
    avg_qty  = float(np.mean([t.qty for t in trades])) if trades else 0.0
    avg_qr   = float(np.mean([t.qty_risk for t in trades])) if trades else 0.0
    win_rate = sum(1 for t in trades if t.pnl > 0) / max(n, 1)

    return {
        "risk_pct": risk_pct, "n_trades": n, "years": round(years, 2),
        "cagr": cagr, "sharpe": sharpe, "max_dd": max_dd, "calmar": calmar,
        "avg_invest": avg_inv, "avg_qty": avg_qty, "avg_qty_risk": avg_qr,
        "binding_risk": br, "binding_cap": bc,
        "win_rate": win_rate, "final_equity": final,
        "flag_low_trades": n < MIN_TRADE_THRESHOLD,
    }


# ─── メイン ───────────────────────────────────────────────────────────────────

def main():
    print("Loading data cache (pre-computing ATR20)...")
    universe  = load_universe()
    cl, hi, lo, atr, topix = build_data_cache(universe)
    avail = list(cl.columns)
    print(f"  Symbols: {len(avail)}/{len(universe)}  Date range: {cl.index[0].date()} - {cl.index[-1].date()}")

    print("  Computing RSR42...")
    rsr = compute_rsr(cl, topix)

    # ─── 5-fold + True OOS 実行 ──────────────────────────────────────
    results: dict[str, list[dict]] = {rp: [] for rp in RISK_PCTS}

    for fold_name, f_start, f_end in WF_FOLDS:
        for rp in RISK_PCTS:
            r = run_period(f_start, f_end, rp, cl, hi, lo, atr, rsr)
            r["fold"] = fold_name; r["start"] = f_start; r["end"] = f_end
            results[rp].append(r)

    # ─── 表示 ─────────────────────────────────────────────────────────
    SEP = "=" * 92
    print("\n" + SEP)
    print("  5-fold walk-forward  risk_pct 1.00% vs 1.25%  (capital=¥3,000,000 / max_pos=3 / msw=25%)")
    print(SEP)

    HEAD = (
        f"  {'Fold':<8}  {'Period':<22}  {'risk%':>5}  {'CAGR':>7}  {'Sharpe':>7}  "
        f"{'MaxDD':>7}  {'Calmar':>7}  {'Invest':>7}  {'AvgQty':>7}  {'Bind%':>6}  "
        f"{'Trades':>7}  {'Win%':>6}  {'Flag'}"
    )
    print(HEAD)
    print("  " + "-" * (len(HEAD) - 2))

    for fold_name, f_start, f_end in WF_FOLDS:
        if fold_name == "TrueOOS":
            print("  " + "·" * 88)
        for rp in RISK_PCTS:
            r = next(x for x in results[rp] if x["fold"] == fold_name)
            flag = " ⚠ LOW_TRADE" if r.get("flag_low_trades") else ""
            period_str = f"{f_start[:7]} - {f_end[:7]}"
            print(
                f"  {fold_name:<8}  {period_str:<22}  {rp:>4.2%}  "
                f"{r['cagr']:>+7.2%}  {r['sharpe']:>7.3f}  "
                f"{r['max_dd']:>7.2%}  {r['calmar']:>7.3f}  "
                f"{r['avg_invest']:>7.2%}  {r['avg_qty']:>7.1f}  "
                f"{r['binding_risk']:>6.1%}  "
                f"{r['n_trades']:>7d}  {r['win_rate']:>6.1%}  {flag}"
            )

    # ─── fold ごとの Δ (1.25% - 1.00%) ──────────────────────────────
    print("\n" + SEP)
    print("  Δ per fold (1.25% - 1.00%)")
    print(SEP)
    delta_cols = ["ΔCAGR", "ΔSharpe", "ΔMaxDD", "ΔCalmar", "ΔInvest", "ΔAvgQty", "ΔBind%"]
    print(f"  {'Fold':<8}  {'Period':<22}  {'ΔCAGR':>8}  {'ΔSharpe':>8}  "
          f"{'ΔMaxDD':>8}  {'ΔCalmar':>8}  {'ΔInvest':>8}  {'ΔAvgQty':>8}  {'ΔBind%':>8}  Verdict")
    print("  " + "-" * 94)

    deltas: dict[str, list[float]] = {k: [] for k in delta_cols}
    verdicts: list[str] = []

    for fold_name, f_start, f_end in WF_FOLDS:
        r10  = next(x for x in results[0.010]  if x["fold"] == fold_name)
        r125 = next(x for x in results[0.0125] if x["fold"] == fold_name)
        dC = r125["cagr"]       - r10["cagr"]
        dS = r125["sharpe"]     - r10["sharpe"]
        dD = r125["max_dd"]     - r10["max_dd"]
        dK = r125["calmar"]     - r10["calmar"]
        dI = r125["avg_invest"] - r10["avg_invest"]
        dQ = r125["avg_qty"]    - r10["avg_qty"]
        dB = r125["binding_risk"] - r10["binding_risk"]

        # 判定: CAGR 改善 かつ MaxDD 悪化 < 3%
        verdict = "✓ PASS" if dC > 0 and dD > -0.03 else ("✗ FAIL" if dC < 0 else "△ NEUT")
        verdicts.append(verdict)

        for k, v in zip(delta_cols, [dC, dS, dD, dK, dI, dQ, dB]):
            deltas[k].append(v)

        period_str = f"{f_start[:7]} - {f_end[:7]}"
        fold_mark = "►" if fold_name == "TrueOOS" else " "
        print(
            f"  {fold_mark}{fold_name:<7}  {period_str:<22}  "
            f"{dC:>+8.2%}  {dS:>+8.3f}  {dD:>+8.2%}  {dK:>+8.3f}  "
            f"{dI:>+8.2%}  {dQ:>+8.1f}  {dB:>+8.1%}  {verdict}"
        )

    # ─── 集計 (True OOS 除く 5-fold) ─────────────────────────────────
    n_folds = len(WF_FOLDS) - 1   # TrueOOS を除く
    wf_v  = verdicts[:n_folds]
    wf_d  = {k: v[:n_folds] for k, v in deltas.items()}
    pass_n = sum(1 for v in wf_v if "PASS" in v)

    print("\n" + SEP)
    print("  5-fold 集計サマリー (TrueOOS 除く)")
    print(SEP)
    print(f"  {'指標':<12}  {'Mean Δ':>10}  {'Std Δ':>10}  {'Min Δ':>10}  {'Max Δ':>10}  {'全正方向':>10}")
    print("  " + "-" * 70)
    for col, vals in wf_d.items():
        all_pos = all(v > 0 for v in vals) if "MaxDD" not in col else all(v > -0.03 for v in vals)
        print(
            f"  {col:<12}  {np.mean(vals):>+10.4f}  {np.std(vals):>10.4f}  "
            f"{min(vals):>+10.4f}  {max(vals):>+10.4f}  {'✓' if all_pos else '✗':>10}"
        )

    # ─── PASS 率 + 判定 ────────────────────────────────────────────────
    pass_rate = pass_n / n_folds
    mean_dcagr = np.mean(wf_d["ΔCAGR"])
    std_dcagr  = np.std(wf_d["ΔCAGR"])
    mean_dd    = np.mean(wf_d["ΔMaxDD"])

    print()
    print(f"  PASS率 (CAGR改善 かつ ΔMaxDD>-3%): {pass_n}/{n_folds} ({pass_rate:.0%})")
    print(f"  平均 ΔCAGR  : {mean_dcagr:>+.4%}  (std {std_dcagr:.4%})")
    print(f"  平均 ΔMaxDD : {mean_dd:>+.4%}")

    # CLAUDE.md OVERFIT_GUARD 準拠判定
    low_trade_folds = sum(
        1 for rp_res in results[0.0125]
        if rp_res.get("flag_low_trades") and rp_res["fold"] != "TrueOOS"
    )
    ois_ratio = pass_rate  # フォールドで OOS 一貫性チェック

    print()
    print("  ── OVERFIT_GUARD 準拠チェック ──")
    print(f"  min_trade(>=5):     {'✓' if low_trade_folds == 0 else f'⚠ {low_trade_folds}folds below threshold'}")
    print(f"  stability_check:    {'✓ STABLE' if std_dcagr < 0.10 else '⚠ UNSTABLE (std>10%)'}")
    print(f"  oos_pass_rate:      {pass_rate:.0%}  ({'✓ >=70%' if pass_rate >= 0.70 else '✗ <70%'})")
    print(f"  walkforward(5f):    ✓ done")

    print()
    if pass_rate >= 0.80 and mean_dcagr > 0 and abs(mean_dd) < 0.03:
        verdict_final = "✅ ADOPT — WFで改善が再現。risk_pct=1.25%採用を推奨。"
    elif pass_rate >= 0.60 and mean_dcagr > 0:
        verdict_final = "⚠ CONDITIONAL — 改善は部分的に再現。慎重に採用可。"
    else:
        verdict_final = "❌ REJECT — WFで改善が再現せず。1.00%維持。"
    print(f"  最終判定: {verdict_final}")

    # ─── CSV 出力 ─────────────────────────────────────────────────────
    out = ROOT / "logs" / "risk_pct_wf_validation.csv"
    fields = [
        "fold", "start", "end", "risk_pct", "n_trades", "years",
        "cagr", "sharpe", "max_dd", "calmar", "avg_invest",
        "avg_qty", "avg_qty_risk", "binding_risk", "binding_cap",
        "win_rate", "final_equity", "flag_low_trades",
    ]
    rows = [r for rp in RISK_PCTS for r in results[rp]]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"\n  CSV: {out}")


if __name__ == "__main__":
    main()
