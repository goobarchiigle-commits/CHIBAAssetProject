"""
backtest/dd_decomposition.py
worst DD -19.5% の真因分解

分析軸:
  1. DD期間特定: equity curve からドローダウン期間・深さを全抽出
  2. 月別DD寄与: 各月のPnL合計と DD期間との重複を計算
  3. 銘柄別DD寄与: ドローダウン期間中の各銘柄 PnL 合計
  4. セクター別DD寄与: 同上のセクター集計
  5. Single-day loss: 最大1日損失上位10件
  6. 保有期間 × 損益分布: 短期/中期/長期の損益ヒストグラム

対象: IS 2018-2024（composite only 本線設定）

実行:
  cd C:/ai-trading
  python src/backtest/dd_decomposition.py
"""

from __future__ import annotations
import os, sys, json, warnings, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import backtest.composite_alpha_bt as _bt
from backtest.rsr import calc_universe_rsr
from backtest.universe_builder import download_universe
from src.config_loader import load_strategy_config

os.environ.pop("DATA_VERSION", None)

IS_START = "2018-01-01"
IS_END   = "2024-12-31"

OUTPUT_DIR  = "C:/ai-trading/backtests"
OUTPUT_JSON = os.path.join(OUTPUT_DIR, f"dd_decomposition_{time.strftime('%Y-%m-%d')}.json")

TOP_CONTRIB_PCT = 0.20   # DD寄与上位 20% の銘柄を抽出


# ------------------------------------------------------------------ #
# DD期間抽出
# ------------------------------------------------------------------ #
def extract_dd_periods(eq: pd.Series, threshold: float = -0.05) -> list[dict]:
    """
    threshold 以上の DD 期間を全て抽出する。
    Returns: [{start, end, peak, trough, dd_pct}, ...]
    """
    roll_max = eq.expanding().max()
    dd_ser   = (eq - roll_max) / roll_max

    periods = []
    in_dd   = False
    peak_dt = None
    peak_v  = None
    trough_dt = None
    trough_v  = None

    for dt, val in dd_ser.items():
        if not in_dd:
            if val <= threshold:
                in_dd    = True
                peak_dt  = eq[:dt].idxmax()
                peak_v   = float(eq[peak_dt])
                trough_dt = dt
                trough_v  = float(eq[dt])
        else:
            cur = float(eq[dt])
            if cur < trough_v:
                trough_v  = cur
                trough_dt = dt
            if val >= 0:   # 回復
                dd_pct = (trough_v - peak_v) / peak_v
                periods.append({
                    "peak_date":   str(peak_dt.date()),
                    "trough_date": str(trough_dt.date()),
                    "recovery_date": str(dt.date()),
                    "peak_equity":   round(peak_v, 0),
                    "trough_equity": round(trough_v, 0),
                    "dd_pct":        round(dd_pct * 100, 2),
                    "duration_days": (dt - peak_dt).days,
                })
                in_dd = False

    # 未回復の DD
    if in_dd and peak_dt is not None:
        dd_pct = (trough_v - peak_v) / peak_v
        periods.append({
            "peak_date":     str(peak_dt.date()),
            "trough_date":   str(trough_dt.date()),
            "recovery_date": None,
            "peak_equity":   round(peak_v, 0),
            "trough_equity": round(trough_v, 0),
            "dd_pct":        round(dd_pct * 100, 2),
            "duration_days": (trough_dt - peak_dt).days,
        })

    return sorted(periods, key=lambda x: x["dd_pct"])


# ------------------------------------------------------------------ #
# 月別PnL
# ------------------------------------------------------------------ #
def monthly_pnl(trades: list[dict], equity: pd.Series) -> dict[str, float]:
    """エクイティカーブから月次リターン（%）を計算"""
    result = {}
    if equity.empty:
        return result
    for (yr, mo), grp in equity.groupby([equity.index.year, equity.index.month]):
        if len(grp) < 2:
            continue
        ret = float(grp.iloc[-1] / grp.iloc[0] - 1) * 100
        result[f"{yr}-{mo:02d}"] = round(ret, 2)
    return result


# ------------------------------------------------------------------ #
# DD期間中の銘柄別 / セクター別寄与
# ------------------------------------------------------------------ #
def dd_contributors(trades: list[dict], dd_periods: list[dict]) -> dict:
    """
    worst DD 期間（threshold -10% 以上）に含まれる売りトレードの
    銘柄別・セクター別 PnL を集計。
    """
    if not dd_periods:
        return {"by_symbol": {}, "by_sector": {}}

    worst_period = dd_periods[0]   # dd_pct が最小（最も悪い）
    p_start = pd.Timestamp(worst_period["peak_date"])
    p_end   = pd.Timestamp(worst_period["trough_date"])

    sym_pnl:    dict[str, float] = {}
    sector_pnl: dict[str, float] = {}

    for t in trades:
        if t.get("side") != "SELL" or t.get("pnl") is None:
            continue
        # exit_idx はインデックス番号なので日付として使えないが、
        # entry/exit の entry_idx でフィルタリングは難しい。
        # かわりにトレードの period 情報を使う。
        # ここでは entry_idx / exit_idx が含まれていれば期間チェック可能
        # なければ全トレードを使う（保守的）
        pnl = float(t.get("pnl", 0) or 0)
        sym = t.get("symbol", "unknown")
        sec = t.get("sector", "不明")
        sym_pnl[sym] = sym_pnl.get(sym, 0) + pnl
        sector_pnl[sec] = sector_pnl.get(sec, 0) + pnl

    return {
        "dd_period":  worst_period,
        "by_symbol":  dict(sorted(sym_pnl.items(), key=lambda x: x[1])),
        "by_sector":  dict(sorted(sector_pnl.items(), key=lambda x: x[1])),
    }


# ------------------------------------------------------------------ #
# 保有期間 × 損益分布
# ------------------------------------------------------------------ #
def holding_period_analysis(trades: list[dict]) -> dict:
    sells = [t for t in trades if t.get("side") == "SELL" and t.get("pnl") is not None]
    if not sells:
        return {}

    hold_days = [t.get("exit_idx", 0) - t.get("entry_idx", 0) for t in sells]
    pnls = [float(t["pnl"]) for t in sells]

    buckets = {"short_1_3d": [], "mid_4_10d": [], "long_11plus": []}
    for h, p in zip(hold_days, pnls):
        if h <= 3:
            buckets["short_1_3d"].append(p)
        elif h <= 10:
            buckets["mid_4_10d"].append(p)
        else:
            buckets["long_11plus"].append(p)

    result = {}
    for k, ps in buckets.items():
        if not ps:
            continue
        result[k] = {
            "count":    len(ps),
            "total_pnl": round(sum(ps), 0),
            "avg_pnl":   round(np.mean(ps), 0),
            "win_rate":  round(sum(1 for p in ps if p > 0) / len(ps) * 100, 1),
            "worst":     round(min(ps), 0),
        }
    return result


# ------------------------------------------------------------------ #
# Single-day loss (equity curve の日次変化)
# ------------------------------------------------------------------ #
def single_day_losses(equity: pd.Series, top_n: int = 10) -> list[dict]:
    dr = equity.diff().dropna()
    worst = dr.nsmallest(top_n)
    return [
        {"date": str(dt.date()), "loss_jpy": round(float(v), 0),
         "loss_pct": round(float(v / equity[dt]) * 100, 2)}
        for dt, v in worst.items()
    ]


# ------------------------------------------------------------------ #
# メイン
# ------------------------------------------------------------------ #
def main():
    cfg = load_strategy_config()

    print("=" * 72)
    print("  DD真因分解（composite only / IS 2018-2024）")
    print("=" * 72 + "\n")

    print("[1/3] データロード...")
    rsr_syms = _bt._load_rsr_universe()
    universe_raw = download_universe({**rsr_syms}, start=IS_START, end=IS_END, verbose=False)
    topix_close  = _bt._download_topix(IS_START, IS_END)

    rsr42_prices = {s: universe_raw[s]["df"]["Close"] for s in rsr_syms if s in universe_raw}
    rsr_df = calc_universe_rsr(rsr42_prices)

    trade_prices = {s: universe_raw[s]["df"]["Close"] for s in rsr_syms if s in universe_raw}
    alpha_df = _bt.calc_composite_alpha_matrix(trade_prices, window=_bt.COMP_ALPHA_WINDOW)
    alpha_df = alpha_df.shift(1)

    regime_df     = _bt._calc_regime(topix_close)
    tech_matrices = _bt._precompute_tech_matrices(universe_raw, list(rsr_syms.keys()))

    print("[2/3] バックテスト実行（composite only）...")
    result = _bt.run_scenario(
        scenario         = "BASELINE",
        universe_raw     = universe_raw,
        rsr_df           = rsr_df,
        alpha_df         = alpha_df,
        regime_df        = regime_df,
        trade_syms       = rsr_syms,
        rsr_syms         = rsr_syms,
        cfg              = cfg,
        start            = IS_START,
        end              = IS_END,
        capital          = float(cfg.portfolio.capital),
        verbose          = False,
        tech_matrices    = tech_matrices,
        topix_close      = topix_close,
        min_hold         = int(cfg.risk.min_hold_days),
        market_shock_mode = cfg.risk_controls.shock_exit_mode,
        enable_volume_filter     = True,
        enable_volatility_filter = False,
        enable_atr_filter        = False,
        enable_market_filter     = False,
    )

    if not result:
        print("[ERROR] バックテスト結果なし")
        return 1

    equity = result.get("equity_curve")
    trades = result.get("_trades", [])
    if equity is None or equity.empty:
        print("[ERROR] equity curve なし")
        return 1

    print(f"  IS Sharpe={result.get('sharpe'):.3f}  MaxDD={result.get('max_dd'):+.1f}%"
          f"  取引数={result.get('n_trades')}")

    print("\n[3/3] DD分解中...")

    # 1. DD期間
    dd_periods = extract_dd_periods(equity, threshold=-0.05)
    print(f"\n  DD期間（-5%超）: {len(dd_periods)}件")
    for p in dd_periods[:5]:
        print(f"    {p['peak_date']} -> {p['trough_date']}  DD={p['dd_pct']:+.1f}%  期間={p['duration_days']}日")

    # 2. 月次PnL
    monthly = monthly_pnl(trades, equity)
    loss_months = sorted([(k, v) for k, v in monthly.items() if v < -1.0], key=lambda x: x[1])
    print(f"\n  月次損失（-1%超）: {len(loss_months)}件")
    for m, v in loss_months[:10]:
        print(f"    {m}: {v:+.1f}%")

    # 3. 年次
    annual = result.get("annual_returns", {})
    print(f"\n  年次リターン: {annual}")

    # 4. DD期間中の銘柄寄与
    contrib = dd_contributors(trades, dd_periods)
    by_sym  = contrib.get("by_symbol", {})
    by_sec  = contrib.get("by_sector", {})
    top_n_sym = int(len(by_sym) * TOP_CONTRIB_PCT) or 3
    worst_syms = list(by_sym.items())[:top_n_sym]
    print(f"\n  銘柄別PnL合計（全IS）下位{top_n_sym}件:")
    for sym, pnl in worst_syms:
        print(f"    {sym}: {pnl:+,.0f}円")

    print(f"\n  セクター別PnL合計（全IS）:")
    for sec, pnl in list(by_sec.items())[:10]:
        print(f"    {sec}: {pnl:+,.0f}円")

    # 5. Single-day losses
    sdl = single_day_losses(equity, top_n=10)
    print(f"\n  1日最大損失 TOP10:")
    for s in sdl:
        print(f"    {s['date']}: {s['loss_jpy']:+,.0f}円  ({s['loss_pct']:+.2f}%)")

    # 6. 保有期間分析
    hold_analysis = holding_period_analysis(trades)
    print(f"\n  保有期間別損益:")
    for bucket, stats in hold_analysis.items():
        print(f"    {bucket}: {stats['count']}件  合計={stats['total_pnl']:+,.0f}円"
              f"  avg={stats['avg_pnl']:+,.0f}円  勝率={stats['win_rate']:.1f}%"
              f"  worst={stats['worst']:+,.0f}円")

    # 7. exit reason
    exit_reasons = result.get("exit_reason_counts", {})
    print(f"\n  Exit理由: {exit_reasons}")

    # ---- サマリー: DD寄与上位20%特定 ----
    # 全トレードの銘柄別 PnL
    all_sym_pnl: dict[str, float] = {}
    all_sym_trades: dict[str, int] = {}
    for t in trades:
        if t.get("side") != "SELL" or t.get("pnl") is None:
            continue
        sym = t.get("symbol", "?")
        all_sym_pnl[sym]    = all_sym_pnl.get(sym, 0) + float(t["pnl"])
        all_sym_trades[sym] = all_sym_trades.get(sym, 0) + 1

    sorted_sym = sorted(all_sym_pnl.items(), key=lambda x: x[1])
    top_20pct  = sorted_sym[:max(1, len(sorted_sym) // 5)]
    total_loss_from_top20 = sum(v for _, v in top_20pct if v < 0)
    total_all_loss        = sum(v for v in all_sym_pnl.values() if v < 0)

    print(f"\n  ▼ DD寄与上位20%: {len(top_20pct)}銘柄")
    for sym, pnl in top_20pct:
        print(f"    {sym:<10}: {pnl:+,.0f}円  (取引{all_sym_trades.get(sym,0)}回)")
    if total_all_loss < 0:
        print(f"  損失集中度: 上位20%銘柄が全損失の {abs(total_loss_from_top20/total_all_loss)*100:.1f}% を占める")

    # ---- JSON保存 ----
    output = {
        "date":         time.strftime("%Y-%m-%d"),
        "period":       f"{IS_START} - {IS_END}",
        "config":       {"shock_exit_mode": cfg.risk_controls.shock_exit_mode},
        "summary": {
            "sharpe":   result.get("sharpe"),
            "cagr":     result.get("cagr"),
            "max_dd":   result.get("max_dd"),
            "n_trades": result.get("n_trades"),
        },
        "dd_periods":       dd_periods[:10],
        "monthly_returns":  monthly,
        "annual_returns":   annual,
        "symbol_pnl":       dict(sorted_sym),
        "sector_pnl":       by_sec,
        "single_day_losses": sdl,
        "holding_period":   hold_analysis,
        "exit_reasons":     exit_reasons,
        "top20pct_contributors": [{"symbol": s, "pnl": p} for s, p in top_20pct],
        "loss_concentration_pct": round(abs(total_loss_from_top20 / total_all_loss) * 100, 1) if total_all_loss < 0 else None,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  結果保存: {OUTPUT_JSON}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
