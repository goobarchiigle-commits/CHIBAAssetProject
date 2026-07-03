"""
backtest/atr_sizing_diagnostic_202606.py
ATR Risk Sizing 詳細診断（WF実行前の原因特定）。推測禁止・実測のみ。
一回限りの監査スクリプト（恒久モジュール化しない）。

期間: 2018-01-01〜2026-06-23（公式PROD_FAITHFULベースライン/risk_pct感度分析と同一、一貫性確保）

実施内容:
  1. APRIL_REPRO（均等配分）と PROD_FAITHFUL(risk_pct=1.25%) の実トレード比較
     （平均/中央値/最大ポジションサイズ、平均エクスポージャー、資金使用率）
  2. APRIL_REPRO実トレードの利益上位50件で「ATR sizingだったら」の counterfactual サイズを算出
  3. 同・損失上位50件
  4. 勝ち/負けトレード群でのATR20・ポジションサイズ・利益率の集計（高ATR=高期待値仮説の検証）
  5. risk_pctレンジ拡張（3.00/3.50/4.00/5.00%、既存0.50-3.00%と合わせて再掲）

実行: python src/backtest/atr_sizing_diagnostic_202606.py
"""
from __future__ import annotations
import os, sys, json, warnings, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import src.backtest.composite_alpha_bt as bt
from src.config_loader import load_strategy_config
from src.strategy.universe import build_dyn_rsr42_active
from src.backtest.universe_builder import download_universe

FULL_START = "2018-01-01"
FULL_END   = "2026-06-23"
OUTPUT_JSON = f"C:/ai-trading/backtests/atr_sizing_diagnostic_202606_{time.strftime('%Y-%m-%d')}.json"


def load_all():
    print("[1/5] ユニバース・価格データ読み込み中（RSR42, 2018-2026）...")
    cfg = load_strategy_config()
    rsr_syms = bt._load_rsr_universe()
    trade_syms = rsr_syms
    all_syms = {**rsr_syms, **trade_syms}

    universe_raw = download_universe(all_syms, start=FULL_START, end=FULL_END, min_days=500, verbose=False)
    topix_close = bt._download_topix(FULL_START, FULL_END)
    print(f"  取得完了: {len(universe_raw)}銘柄, TOPIX={len(topix_close)}日")

    print("[2/5] RSR計算中...")
    rsr42_prices = {s: universe_raw[s]["df"]["Close"] for s in rsr_syms if s in universe_raw}
    rsr_df = bt.calc_universe_rsr(rsr42_prices)

    print("[3/5] Composite Alpha計算中...")
    trade_prices = {s: universe_raw[s]["df"]["Close"] for s in trade_syms if s in universe_raw}
    alpha_df = bt.calc_composite_alpha_matrix(trade_prices, window=bt.COMP_ALPHA_WINDOW)
    alpha_df = alpha_df.shift(1)

    regime_df = bt._calc_regime(topix_close)

    print("[4/5] 動的ユニバース(dyn_rsr42_bear_rs0)構築中...")
    bear_filter_cfg = cfg.risk_controls.bear_universe_filter
    bear_exclude = list(bear_filter_cfg.excluded_sectors) if bear_filter_cfg.enabled else None
    sym_active_df = build_dyn_rsr42_active(
        universe_raw=universe_raw, topix_close=topix_close, rsr_df=rsr_df,
        all_syms=list(trade_syms.keys()), start=FULL_START, end=FULL_END,
        bear_exclude_sectors=bear_exclude,
        sym_sector_map=dict(trade_syms) if bear_exclude else None,
    )

    print("[5/5] テクニカル指標事前計算中...")
    tech_matrices = bt._precompute_tech_matrices(universe_raw, list(trade_syms.keys()))

    return cfg, universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms, sym_active_df, tech_matrices


def run_config(cfg, universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms,
                tech_matrices, *, use_dyn, shock_mode, rsr_exit, atr_trail, ml, sizing, mtf, risk_pct=None):
    """use_dyn: None=動的ユニバース無効 / sym_active_df本体=有効"""
    return bt.run_scenario(
        scenario="PROD_FAITHFUL" if (atr_trail or ml or sizing or mtf) else "BASELINE",
        universe_raw=universe_raw, rsr_df=rsr_df, alpha_df=alpha_df, regime_df=regime_df,
        trade_syms=trade_syms, rsr_syms=rsr_syms, cfg=cfg,
        start=FULL_START, end=FULL_END, verbose=False,
        tech_matrices=tech_matrices,
        capital=cfg.portfolio.capital,
        min_hold=cfg.risk.min_hold_days,
        market_shock_mode=shock_mode,
        rsr_exit_threshold=rsr_exit,
        sym_active_df=use_dyn,
        enable_atr_trailing_prod=atr_trail,
        enable_multilayer_rsr=ml,
        enable_atr_risk_sizing=sizing,
        enable_mtf_filter=mtf,
        risk_sizing_pct=(risk_pct if risk_pct is not None else bt.PROD_RISK_PCT),
    )


def atr_at_signal_day(tech_matrices, common_dates, symbol, entry_idx):
    """entry_idx（next_iで記録された執行日インデックス）から1日前（シグナル日）のATR20を取得。"""
    sig_idx = entry_idx - 1
    if sig_idx < 0 or sig_idx >= len(common_dates):
        return float("nan")
    date = common_dates[sig_idx]
    atr_df = tech_matrices["atr20"]
    if date not in atr_df.index or symbol not in atr_df.columns:
        return float("nan")
    val = atr_df.loc[date, symbol]
    return float(val) if pd.notna(val) else float("nan")


def counterfactual_atr_qty(capital, risk_pct, atr_val, entry_price, max_single_weight, lot=100):
    if np.isnan(atr_val) or atr_val <= 0:
        return None  # ATR取得不能（counterfactual算出不可）
    risk_per_trade = capital * risk_pct
    qty_raw = int(risk_per_trade / atr_val)
    qty_risk = (qty_raw // lot) * lot
    if qty_risk == 0:
        qty_risk = lot
    qty_cap = int((capital * max_single_weight) // (entry_price * lot)) * lot
    qty = min(qty_risk, qty_cap)
    return max(qty, 0)


def main():
    cfg, universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms, sym_active_df, tech_matrices = load_all()
    capital = cfg.portfolio.capital
    max_sw = cfg.portfolio.max_single_weight
    common_dates_ref = None

    print(f"\n{'='*78}\n  [Part0] APRIL_REPRO / PROD_FAITHFUL(1.25%) 実行（期間: {FULL_START}〜{FULL_END}）\n{'='*78}")

    res_april = run_config(
        cfg, universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms, tech_matrices,
        use_dyn=None, shock_mode="full_exit", rsr_exit=75.0,
        atr_trail=False, ml=False, sizing=False, mtf=False,
    )
    res_prod = run_config(
        cfg, universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms, tech_matrices,
        use_dyn=sym_active_df, shock_mode="composite", rsr_exit=70.0,
        atr_trail=True, ml=True, sizing=True, mtf=True, risk_pct=0.0125,
    )

    print(f"  APRIL_REPRO     : CAGR={res_april['cagr']:+.1f}%  Sharpe={res_april['sharpe']:.3f}  "
          f"trades={res_april['n_trades']}  avgExp={res_april['avg_exposure']:.1f}%")
    print(f"  PROD_FAITHFUL   : CAGR={res_prod['cagr']:+.1f}%  Sharpe={res_prod['sharpe']:.3f}  "
          f"trades={res_prod['n_trades']}  avgExp={res_prod['avg_exposure']:.1f}%")

    # 共通の日付インデックス（ATR lookup用） — run_scenario内で再構築されたcommon_datesと一致させるため、
    # universe_rawとtrade_symsから同一ロジックで再計算する。
    active_syms = [s for s in trade_syms if s in universe_raw]
    common_dates = None
    for s in active_syms:
        idx = universe_raw[s]["df"].index
        common_dates = idx if common_dates is None else common_dates.intersection(idx)
    common_dates = common_dates.sort_values()
    ts_f, te_f = pd.Timestamp(FULL_START), pd.Timestamp(FULL_END)
    common_dates = common_dates[(common_dates >= ts_f) & (common_dates <= te_f)]

    april_trades = res_april["_trades"]
    prod_trades = res_prod["_trades"]

    # ---------------------------------------------------------------- #
    # [1] 全トレード比較（ポジションサイズ統計）
    # ---------------------------------------------------------------- #
    def position_size_stats(trades):
        sizes = np.array([t["qty"] * t["entry"] for t in trades if t["entry"] and t["qty"]])
        return {
            "n": len(sizes),
            "mean_size": float(np.mean(sizes)) if len(sizes) else 0.0,
            "median_size": float(np.median(sizes)) if len(sizes) else 0.0,
            "max_size": float(np.max(sizes)) if len(sizes) else 0.0,
            "mean_size_pct_of_capital": float(np.mean(sizes) / capital * 100) if len(sizes) else 0.0,
        }

    stats_april = position_size_stats(april_trades)
    stats_prod = position_size_stats(prod_trades)

    print(f"\n{'='*78}\n  [1] ポジションサイズ統計比較\n{'='*78}")
    print(f"  {'指標':30s}  {'APRIL_REPRO':>15}  {'PROD_FAITHFUL':>15}")
    print(f"  {'平均ポジションサイズ(円)':30s}  {stats_april['mean_size']:>15,.0f}  {stats_prod['mean_size']:>15,.0f}")
    print(f"  {'中央値ポジションサイズ(円)':30s}  {stats_april['median_size']:>15,.0f}  {stats_prod['median_size']:>15,.0f}")
    print(f"  {'最大ポジションサイズ(円)':30s}  {stats_april['max_size']:>15,.0f}  {stats_prod['max_size']:>15,.0f}")
    print(f"  {'平均エクスポージャー(%)':30s}  {res_april['avg_exposure']:>15.1f}  {res_prod['avg_exposure']:>15.1f}")
    print(f"  {'平均サイズ/資本(%)':30s}  {stats_april['mean_size_pct_of_capital']:>15.1f}  {stats_prod['mean_size_pct_of_capital']:>15.1f}")

    # ---------------------------------------------------------------- #
    # [2][3][4] APRIL_REPRO実トレードに ATR sizing counterfactual を付与
    # ---------------------------------------------------------------- #
    enriched = []
    for t in april_trades:
        if not t.get("entry") or not t.get("qty"):
            continue
        sym = t["symbol"]
        entry_idx = t["entry_idx"]
        entry_price = t["entry"]
        return_pct = (t["exit"] / t["entry"] - 1) * 100 if t["exit"] else None
        atr_val = atr_at_signal_day(tech_matrices, common_dates, sym, entry_idx)
        april_size = t["qty"] * entry_price
        atr_qty = counterfactual_atr_qty(capital, 0.0125, atr_val, entry_price, max_sw)
        atr_size = (atr_qty * entry_price) if atr_qty is not None else None
        size_reduction_pct = (
            (1 - atr_size / april_size) * 100 if (atr_size is not None and april_size > 0) else None
        )
        entry_date = str(common_dates[entry_idx].date()) if 0 <= entry_idx < len(common_dates) else None
        atr_pct = (atr_val / entry_price * 100) if (not np.isnan(atr_val) and entry_price > 0) else float("nan")
        enriched.append({
            "symbol": sym, "entry_date": entry_date, "return_pct": return_pct,
            "pnl": t["pnl"], "entry_price": entry_price, "atr20_at_entry": atr_val, "atr_pct": atr_pct,
            "april_size": april_size, "atr_size": atr_size,
            "size_reduction_pct": size_reduction_pct, "atr_qty": atr_qty,
        })

    valid = [e for e in enriched if e["return_pct"] is not None and e["atr_size"] is not None]
    winners = sorted(valid, key=lambda e: -e["return_pct"])[:50]
    losers = sorted(valid, key=lambda e: e["return_pct"])[:50]

    print(f"\n{'='*78}\n  [2] 利益上位50トレード: APRIL size vs ATR sizing counterfactual\n{'='*78}")
    print(f"  {'symbol':10s} {'entry_date':12s} {'ret%':>7s} {'APRIL(円)':>12s} {'ATR(円)':>12s} {'縮小%':>7s}")
    for e in winners[:50]:
        print(f"  {e['symbol']:10s} {e['entry_date']:12s} {e['return_pct']:+6.1f}% "
              f"{e['april_size']:>12,.0f} {e['atr_size']:>12,.0f} {e['size_reduction_pct']:>6.1f}%")

    print(f"\n{'='*78}\n  [3] 損失上位50トレード: APRIL size vs ATR sizing counterfactual\n{'='*78}")
    print(f"  {'symbol':10s} {'entry_date':12s} {'ret%':>7s} {'APRIL(円)':>12s} {'ATR(円)':>12s} {'縮小%':>7s}")
    for e in losers[:50]:
        print(f"  {e['symbol']:10s} {e['entry_date']:12s} {e['return_pct']:+6.1f}% "
              f"{e['april_size']:>12,.0f} {e['atr_size']:>12,.0f} {e['size_reduction_pct']:>6.1f}%")

    # ---------------------------------------------------------------- #
    # [4] 勝ち/負けトレード群集計
    # ---------------------------------------------------------------- #
    win_group = [e for e in valid if e["pnl"] is not None and e["pnl"] > 0]
    lose_group = [e for e in valid if e["pnl"] is not None and e["pnl"] <= 0]

    def group_summary(group):
        if not group:
            return {}
        return {
            "n": len(group),
            "avg_atr20_yen": float(np.mean([g["atr20_at_entry"] for g in group if not np.isnan(g["atr20_at_entry"])])),
            "avg_atr_pct": float(np.mean([g["atr_pct"] for g in group if not np.isnan(g["atr_pct"])])),
            "avg_entry_price": float(np.mean([g["entry_price"] for g in group])),
            "avg_april_size": float(np.mean([g["april_size"] for g in group])),
            "avg_atr_size": float(np.mean([g["atr_size"] for g in group])),
            "avg_size_reduction_pct": float(np.mean([g["size_reduction_pct"] for g in group])),
            "avg_return_pct": float(np.mean([g["return_pct"] for g in group])),
        }

    summary_win = group_summary(win_group)
    summary_lose = group_summary(lose_group)

    print(f"\n{'='*78}\n  [4] 勝ち/負けトレード群 集計（高ATR=高期待値仮説の検証）\n{'='*78}")
    print(f"  {'指標':25s}  {'勝ちトレード':>15}  {'負けトレード':>15}")
    print(f"  {'件数':25s}  {summary_win.get('n',0):>15}  {summary_lose.get('n',0):>15}")
    print(f"  {'平均ATR20(円)':25s}  {summary_win.get('avg_atr20_yen',0):>15.1f}  {summary_lose.get('avg_atr20_yen',0):>15.1f}")
    print(f"  {'平均ATR%(ATR/価格)':25s}  {summary_win.get('avg_atr_pct',0):>14.2f}%  {summary_lose.get('avg_atr_pct',0):>14.2f}%")
    print(f"  {'平均エントリー価格(円)':25s}  {summary_win.get('avg_entry_price',0):>15,.0f}  {summary_lose.get('avg_entry_price',0):>15,.0f}")
    print(f"  {'平均APRILサイズ(円)':25s}  {summary_win.get('avg_april_size',0):>15,.0f}  {summary_lose.get('avg_april_size',0):>15,.0f}")
    print(f"  {'平均ATRサイズ(円)':25s}  {summary_win.get('avg_atr_size',0):>15,.0f}  {summary_lose.get('avg_atr_size',0):>15,.0f}")
    print(f"  {'平均サイズ縮小率(%)':25s}  {summary_win.get('avg_size_reduction_pct',0):>15.1f}  {summary_lose.get('avg_size_reduction_pct',0):>15.1f}")
    print(f"  {'平均利益率(%)':25s}  {summary_win.get('avg_return_pct',0):>15.1f}  {summary_lose.get('avg_return_pct',0):>15.1f}")

    # ---------------------------------------------------------------- #
    # [5] risk_pctレンジ拡張
    # ---------------------------------------------------------------- #
    print(f"\n{'='*78}\n  [5] risk_pctレンジ拡張（3.00/3.50/4.00/5.00%）\n{'='*78}")
    extra_rows = []
    for risk_pct in [0.0300, 0.0350, 0.0400, 0.0500]:
        res = run_config(
            cfg, universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms, tech_matrices,
            use_dyn=sym_active_df, shock_mode="composite", rsr_exit=70.0,
            atr_trail=True, ml=True, sizing=True, mtf=True, risk_pct=risk_pct,
        )
        row = {
            "risk_pct": risk_pct, "cagr": res["cagr"], "sharpe": res["sharpe"], "max_dd": res["max_dd"],
            "calmar": res["calmar"], "profit_factor": res["profit_factor"], "avg_exposure": res["avg_exposure"],
            "n_trades": res["n_trades"],
        }
        extra_rows.append(row)
        print(f"  risk_pct={risk_pct*100:5.2f}%  CAGR={row['cagr']:+6.1f}%  Sharpe={row['sharpe']:.3f}  "
              f"MaxDD={row['max_dd']:+6.1f}%  Calmar={row['calmar']:.3f}  PF={row['profit_factor']:.3f}  "
              f"avgExp={row['avg_exposure']:.1f}%")

    # ---------------------------------------------------------------- #
    # 保存
    # ---------------------------------------------------------------- #
    out = {
        "period": [FULL_START, FULL_END],
        "april_repro_summary": {k: res_april[k] for k in ("cagr", "sharpe", "max_dd", "calmar", "profit_factor", "n_trades", "avg_exposure")},
        "prod_faithful_125_summary": {k: res_prod[k] for k in ("cagr", "sharpe", "max_dd", "calmar", "profit_factor", "n_trades", "avg_exposure")},
        "position_size_stats": {"april_repro": stats_april, "prod_faithful": stats_prod},
        "top50_winners": winners,
        "top50_losers": losers,
        "win_group_summary": summary_win,
        "lose_group_summary": summary_lose,
        "risk_pct_extended": extra_rows,
    }
    os.makedirs("C:/ai-trading/backtests", exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n結果保存: {OUTPUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
