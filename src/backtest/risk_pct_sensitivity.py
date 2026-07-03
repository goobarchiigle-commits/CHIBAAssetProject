"""
risk_pct_sensitivity.py  —  ATR ベース risk_pct IS/OOS 感度分析

risk_pct = 1.0%, 1.25%, 1.5%, 2.0% の各設定について
IS (2018–2024) / OOS (2025) で CAGR / Sharpe / MaxDD / Calmar /
平均投資率 / qty_risk binding 率 を計算し比較する。

シグナルロジック（signal_bridge.py と同等）:
  Entry  : RSR42 >= 75 かつ 20 日高値ブレイク
  Exit   : 55 日安値割れ（タートル）OR 60 日タイムストップ
  Sizing : qty = min(qty_risk, qty_cap)
           qty_risk = int(capital × risk_pct / ATR20) // LOT × LOT
           qty_cap  = int(capital × max_single_weight / lot_cost) // LOT × LOT
"""
from __future__ import annotations

import sys, json
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

# ─── パス ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]
BD_DIR     = ROOT / "data" / "backtest_dataset" / "2026-03-28"
UNI_FILE   = ROOT / "src" / "configs" / "universe" / "rsr42_v1_gov2.json"
TOPIX_SYM  = "1306.T"          # TOPIX ETF として代用

# ─── 定数 ────────────────────────────────────────────────────────────────────
CAPITAL           = 3_000_000
MAX_POSITIONS     = 3
MAX_SINGLE_WEIGHT = 0.25
MAX_HOLD_DAYS     = 60
LOT               = 100
COST_ONE_WAY      = 0.001 + 0.00055   # slippage + commission
MIN_RSR           = 75.0
TURTLE_ENTRY_N    = 20
TURTLE_EXIT_N     = 55

IS_START, IS_END  = "2018-01-01", "2024-12-31"
OOS_START, OOS_END = "2025-01-01", "2025-12-31"

RISK_PCTS = [0.010, 0.0125, 0.015, 0.020]

# ─── データロード ─────────────────────────────────────────────────────────────

def load_universe() -> dict[str, str]:
    if UNI_FILE.exists():
        d = json.loads(UNI_FILE.read_text(encoding="utf-8"))
        return d.get("symbols", d)
    # fallback: live universe
    from src.paths import LIVE_UNIVERSE_FILE
    d = json.loads(Path(LIVE_UNIVERSE_FILE).read_text(encoding="utf-8"))
    return d.get("symbols", d)


def load_ohlcv(sym: str) -> pd.DataFrame | None:
    p = BD_DIR / f"{sym}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    df = df[[close_col, "High", "Low", "Volume"]].copy()
    df.columns = ["Close", "High", "Low", "Volume"]
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(None).normalize()
    df = df[~df.index.duplicated()].sort_index()
    return df


def build_panel(universe: dict[str, str]) -> pd.DataFrame:
    """Close 価格パネル {symbol: Series} を構築する。"""
    frames = {}
    for sym in universe:
        df = load_ohlcv(sym)
        if df is not None and len(df) > 100:
            frames[sym] = df["Close"]
    return pd.DataFrame(frames).sort_index()


def compute_rsr(panel: pd.DataFrame, topix: pd.Series, period: int = 42) -> pd.DataFrame:
    """RSR42: 各銘柄の period 日リターン − TOPIX 同期間リターン の cross-sectional 百分位。"""
    sym_ret   = panel.pct_change(period)
    topix_ret = topix.pct_change(period)
    excess    = sym_ret.sub(topix_ret, axis=0)
    # 百分位 rank (0-100) — cross-sectional
    rsr = excess.rank(axis=1, pct=True) * 100
    return rsr


# ─── バックテスト ─────────────────────────────────────────────────────────────

@dataclass
class Trade:
    symbol:      str
    entry_date:  pd.Timestamp
    entry_price: float
    qty:         int
    entry_atr:   float
    qty_risk:    int
    qty_cap:     int
    binding:     str    # "qty_risk" | "qty_cap" | "equal"
    exit_date:   pd.Timestamp | None = None
    exit_price:  float | None = None
    pnl:         float = 0.0


def _atr20(hi: pd.Series, lo: pd.Series, cl: pd.Series) -> float:
    """最新 ATR20 を計算する。"""
    tr = pd.concat([
        hi - lo,
        (hi - cl.shift(1)).abs(),
        (lo - cl.shift(1)).abs(),
    ], axis=1).max(axis=1)
    v = tr.rolling(20).mean().dropna()
    return float(v.iloc[-1]) if not v.empty else 0.0


def run_backtest(
    start:    str,
    end:      str,
    risk_pct: float,
    panel:    pd.DataFrame,
    highs:    dict[str, pd.Series],
    lows:     dict[str, pd.Series],
    rsr:      pd.DataFrame,
    verbose:  bool = False,
) -> dict:
    dates = panel.loc[start:end].index
    if len(dates) < 60:
        return {}

    cash       = float(CAPITAL)
    peak_eq    = float(CAPITAL)
    positions: dict[str, Trade] = {}
    equity_log: list[float]     = []
    trades: list[Trade]         = []
    daily_invest: list[float]   = []

    position_budget = CAPITAL * MAX_SINGLE_WEIGHT
    risk_budget     = CAPITAL * risk_pct

    for i, date in enumerate(dates):
        # 終値
        closes = panel.loc[date]

        # ポジション評価額
        port_value = sum(
            t.qty * float(closes.get(sym, t.entry_price))
            for sym, t in positions.items()
        )
        equity = cash + port_value
        peak_eq = max(peak_eq, equity)
        equity_log.append(equity)
        daily_invest.append(port_value / max(equity, 1))

        # ── Exit: turtle 55 日安値 OR タイムストップ ─────────────────
        to_exit = []
        for sym, t in positions.items():
            if sym not in panel.columns or date not in panel.index:
                continue
            price = float(closes.get(sym, t.entry_price))
            held  = i - dates.get_loc(t.entry_date)
            # 55 日安値
            sym_lows = lows.get(sym)
            if sym_lows is not None:
                past_lows = sym_lows.loc[:date].iloc[-TURTLE_EXIT_N-1:-1]
                exit_stop = float(past_lows.min()) if not past_lows.empty else 0.0
            else:
                exit_stop = 0.0
            if price <= exit_stop or held >= MAX_HOLD_DAYS:
                to_exit.append(sym)

        for sym in to_exit:
            t = positions.pop(sym)
            price = float(closes.get(sym, t.entry_price))
            proceeds = t.qty * price * (1 - COST_ONE_WAY)
            cash += proceeds
            t.exit_date  = date
            t.exit_price = price
            t.pnl        = (price - t.entry_price) * t.qty
            trades.append(t)

        # ── Entry: RSR >= 75 + 20 日高値ブレイク ───────────────────
        if len(positions) >= MAX_POSITIONS:
            continue

        if date not in rsr.index:
            continue
        rsr_today = rsr.loc[date].dropna()
        candidates = rsr_today[rsr_today >= MIN_RSR].sort_values(ascending=False)

        open_slots = MAX_POSITIONS - len(positions)
        new_today  = 0

        for sym in candidates.index:
            if open_slots <= 0 or new_today >= 1:
                break
            if sym in positions:
                continue
            if sym not in panel.columns:
                continue

            price = float(closes.get(sym, np.nan))
            if np.isnan(price) or price <= 0:
                continue

            # 20 日高値ブレイク確認
            sym_highs = highs.get(sym)
            if sym_highs is None:
                continue
            past_hi = sym_highs.loc[:date].iloc[-TURTLE_ENTRY_N-1:-1]
            if past_hi.empty:
                continue
            if price <= float(past_hi.max()):
                continue

            # ATR20
            df_sym = load_ohlcv(sym)
            if df_sym is None:
                continue
            hist = df_sym.loc[:date].tail(25)
            if len(hist) < 21:
                continue
            atr = _atr20(hist["High"], hist["Low"], hist["Close"])
            if atr <= 0:
                continue

            lot_cost = price * LOT

            # qty_risk
            qty_raw  = int(risk_budget / atr)
            qty_risk = (qty_raw // LOT) * LOT
            if qty_risk == 0:
                qty_risk = LOT

            # qty_cap
            qty_cap = int(position_budget // lot_cost) * LOT
            if qty_cap <= 0:
                continue

            # final
            qty = min(qty_risk, qty_cap)
            if qty <= 0:
                continue

            cost_yen = qty * price * (1 + COST_ONE_WAY)
            if cash < cost_yen:
                continue

            if qty_risk < qty_cap:
                binding = "qty_risk"
            elif qty_cap < qty_risk:
                binding = "qty_cap"
            else:
                binding = "equal"

            cash -= cost_yen
            t = Trade(
                symbol=sym, entry_date=date, entry_price=price,
                qty=qty, entry_atr=atr,
                qty_risk=qty_risk, qty_cap=qty_cap, binding=binding,
            )
            positions[sym] = t
            open_slots -= 1
            new_today  += 1

    # 残ポジション強制決済
    final_date = dates[-1]
    for sym, t in positions.items():
        price = float(panel.loc[final_date].get(sym, t.entry_price))
        proceeds = t.qty * price * (1 - COST_ONE_WAY)
        cash += proceeds
        t.exit_date  = final_date
        t.exit_price = price
        t.pnl        = (price - t.entry_price) * t.qty
        trades.append(t)

    # ─── メトリクス計算 ───────────────────────────────────────────────
    eq = pd.Series(equity_log, index=dates)
    years = len(dates) / 252
    final_eq = eq.iloc[-1]

    cagr   = (final_eq / CAPITAL) ** (1 / max(years, 0.01)) - 1
    dd     = (eq / eq.cummax() - 1)
    max_dd = float(dd.min())

    daily_ret = eq.pct_change().dropna()
    sharpe    = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0.0
    calmar    = cagr / abs(max_dd) if max_dd < 0 else 0.0

    avg_invest = float(np.mean(daily_invest)) if daily_invest else 0.0
    n_trades   = len(trades)
    binding_risk = sum(1 for t in trades if t.binding == "qty_risk")
    binding_cap  = sum(1 for t in trades if t.binding == "qty_cap")
    binding_rate = binding_risk / max(n_trades, 1)

    win_rate = sum(1 for t in trades if t.pnl > 0) / max(n_trades, 1)
    avg_qty_risk = np.mean([t.qty_risk for t in trades]) if trades else 0
    avg_qty_cap  = np.mean([t.qty_cap  for t in trades]) if trades else 0
    avg_qty_fin  = np.mean([t.qty      for t in trades]) if trades else 0

    return {
        "risk_pct":        risk_pct,
        "cagr":            cagr,
        "sharpe":          sharpe,
        "max_dd":          max_dd,
        "calmar":          calmar,
        "avg_invest":      avg_invest,
        "binding_risk_rate": binding_rate,
        "binding_cap_rate":  binding_cap / max(n_trades, 1),
        "n_trades":        n_trades,
        "win_rate":        win_rate,
        "avg_qty_risk":    avg_qty_risk,
        "avg_qty_cap":     avg_qty_cap,
        "avg_qty_final":   avg_qty_fin,
        "final_equity":    final_eq,
        "years":           years,
    }


# ─── メイン ───────────────────────────────────────────────────────────────────

def main():
    print("Loading universe and OHLCV data...")
    universe = load_universe()
    panel    = build_panel(universe)
    avail    = list(panel.columns)
    print(f"  Symbols with data: {len(avail)}/{len(universe)}")

    # TOPIX
    topix_df = load_ohlcv(TOPIX_SYM)
    if topix_df is None:
        raise RuntimeError(f"TOPIX ({TOPIX_SYM}) not found in dataset")
    topix = topix_df["Close"].reindex(panel.index).ffill()

    # RSR
    print("  Computing RSR42...")
    rsr = compute_rsr(panel, topix, period=42)

    # High / Low キャッシュ（毎回 load_ohlcv を呼ぶ代わり）
    highs: dict[str, pd.Series] = {}
    lows:  dict[str, pd.Series] = {}
    for sym in avail:
        df = load_ohlcv(sym)
        if df is not None:
            highs[sym] = df["High"].reindex(panel.index).ffill()
            lows[sym]  = df["Low"].reindex(panel.index).ffill()

    # ─── IS / OOS × risk_pct スイープ ─────────────────────────────────
    print("\n" + "=" * 80)
    print("  risk_pct IS/OOS 感度分析  (capital=¥3,000,000 / max_pos=3 / msw=25%)")
    print("=" * 80)

    header = (
        f"  {'Period':<6}  {'risk_pct':>9}  {'CAGR':>7}  {'Sharpe':>7}  "
        f"{'MaxDD':>7}  {'Calmar':>7}  {'AvgInvst':>9}  {'qrisk_bind':>11}  "
        f"{'n_trades':>9}  {'WinRate':>8}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    all_results: dict[str, list[dict]] = {"IS": [], "OOS": []}

    for risk_pct in RISK_PCTS:
        for period, s, e in [("IS", IS_START, IS_END), ("OOS", OOS_START, OOS_END)]:
            r = run_backtest(
                start=s, end=e, risk_pct=risk_pct,
                panel=panel, highs=highs, lows=lows, rsr=rsr,
            )
            if not r:
                continue
            r["period"] = period
            all_results[period].append(r)

            print(
                f"  {period:<6}  {risk_pct:>8.2%}  "
                f"{r['cagr']:>+7.2%}  {r['sharpe']:>7.3f}  "
                f"{r['max_dd']:>7.2%}  {r['calmar']:>7.3f}  "
                f"{r['avg_invest']:>9.2%}  "
                f"{r['binding_risk_rate']:>11.2%}  "
                f"{r['n_trades']:>9d}  "
                f"{r['win_rate']:>8.2%}"
            )

    # ─── 変化量テーブル ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  Δ vs risk_pct=1.0% (OOS)")
    print("=" * 80)
    oos_base = next((r for r in all_results["OOS"] if abs(r["risk_pct"] - 0.01) < 1e-5), None)
    if oos_base:
        print(f"  {'risk_pct':>9}  {'ΔCAGR':>8}  {'ΔSharpe':>8}  {'ΔMaxDD':>8}  {'ΔCalmar':>8}  {'ΔInvst':>8}  {'ΔBindRisk':>10}")
        print("  " + "-" * 72)
        for r in all_results["OOS"]:
            rp = r["risk_pct"]
            print(
                f"  {rp:>8.2%}  "
                f"{(r['cagr']    - oos_base['cagr']):>+8.2%}  "
                f"{(r['sharpe']  - oos_base['sharpe']):>+8.3f}  "
                f"{(r['max_dd']  - oos_base['max_dd']):>+8.2%}  "
                f"{(r['calmar']  - oos_base['calmar']):>+8.3f}  "
                f"{(r['avg_invest'] - oos_base['avg_invest']):>+8.2%}  "
                f"{(r['binding_risk_rate'] - oos_base['binding_risk_rate']):>+10.2%}"
            )

    # ─── qty detail ───────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  OOS 平均ロットサイズ詳細")
    print("=" * 80)
    print(f"  {'risk_pct':>9}  {'avg_qty_risk':>13}  {'avg_qty_cap':>12}  {'avg_qty_final':>14}  {'invest/pos':>11}")
    print("  " + "-" * 66)
    for r in all_results["OOS"]:
        invest_per_pos = r["avg_qty_final"] * (CAPITAL * MAX_SINGLE_WEIGHT / (MAX_POSITIONS * 100))
        print(
            f"  {r['risk_pct']:>8.2%}  "
            f"{r['avg_qty_risk']:>13.1f}  "
            f"{r['avg_qty_cap']:>12.1f}  "
            f"{r['avg_qty_final']:>14.1f}  "
        )

    # ─── IS/OOS 乖離（過学習チェック） ────────────────────────────────
    print("\n" + "=" * 80)
    print("  IS/OOS 乖離 (OOS degradation = OOS_CAGR / IS_CAGR)")
    print("=" * 80)
    print(f"  {'risk_pct':>9}  {'IS_CAGR':>9}  {'OOS_CAGR':>10}  {'OOS/IS':>8}  {'IS_Calmar':>10}  {'OOS_Calmar':>11}")
    print("  " + "-" * 66)
    for rp in RISK_PCTS:
        ri = next((r for r in all_results["IS"]  if abs(r["risk_pct"] - rp) < 1e-5), None)
        ro = next((r for r in all_results["OOS"] if abs(r["risk_pct"] - rp) < 1e-5), None)
        if ri and ro:
            ratio = ro["cagr"] / ri["cagr"] if abs(ri["cagr"]) > 1e-6 else 0
            print(
                f"  {rp:>8.2%}  "
                f"{ri['cagr']:>+9.2%}  {ro['cagr']:>+10.2%}  "
                f"{ratio:>8.3f}  "
                f"{ri['calmar']:>10.3f}  {ro['calmar']:>11.3f}"
            )

    # CSV 出力
    out_path = ROOT / "logs" / "risk_pct_sensitivity.csv"
    out_path.parent.mkdir(exist_ok=True)
    import csv
    fields = [
        "period", "risk_pct", "cagr", "sharpe", "max_dd", "calmar",
        "avg_invest", "binding_risk_rate", "binding_cap_rate",
        "n_trades", "win_rate", "avg_qty_risk", "avg_qty_cap", "avg_qty_final",
        "final_equity", "years",
    ]
    rows = all_results["IS"] + all_results["OOS"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n  CSV: {out_path}")


if __name__ == "__main__":
    main()
