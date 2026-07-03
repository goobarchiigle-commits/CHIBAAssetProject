"""
backtest/risk_pct_sensitivity_202606.py
ATR Risk Sizing の risk_pct 感度分析（PROD_FAITHFUL固定、risk_pctのみ変動）。
推測禁止・実測のみ。一回限りの監査スクリプト（恒久モジュール化しない）。

固定: Dynamic Universe ON / Concentration Caps ON / RSR Exit=70 / ATR Trailing ON /
      MultiLayer RSR ON / MTF Filter ON（=PROD_FAITHFUL全機能）
変動: risk_pct = capital×risk_pct÷ATR20 のrisk_pctのみ
期間: 2018-01-01〜2026-06-23（公式PROD_FAITHFULベースラインと同一期間）

実行: python src/backtest/risk_pct_sensitivity_202606.py
"""
from __future__ import annotations
import os, sys, json, warnings, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import src.backtest.composite_alpha_bt as bt
from src.config_loader import load_strategy_config
from src.strategy.universe import build_dyn_rsr42_active
from src.backtest.universe_builder import download_universe

FULL_START = "2018-01-01"
FULL_END   = "2026-06-23"
RISK_PCT_VALUES = [0.0050, 0.0075, 0.0100, 0.0125, 0.0150, 0.0175, 0.0200, 0.0250, 0.0300]

OUTPUT_JSON = f"C:/ai-trading/backtests/risk_pct_sensitivity_202606_{time.strftime('%Y-%m-%d')}.json"


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


def main():
    cfg, universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms, sym_active_df, tech_matrices = load_all()

    rows = []
    print(f"\n{'='*78}\n  ATR Risk Sizing 感度分析（PROD_FAITHFUL固定, risk_pctのみ変動）\n  期間: {FULL_START}〜{FULL_END}\n{'='*78}")
    for risk_pct in RISK_PCT_VALUES:
        res = bt.run_scenario(
            scenario="PROD_FAITHFUL",
            universe_raw=universe_raw, rsr_df=rsr_df, alpha_df=alpha_df, regime_df=regime_df,
            trade_syms=trade_syms, rsr_syms=rsr_syms, cfg=cfg,
            start=FULL_START, end=FULL_END, verbose=False,
            tech_matrices=tech_matrices,
            capital=cfg.portfolio.capital,
            min_hold=cfg.risk.min_hold_days,
            market_shock_mode="composite",
            rsr_exit_threshold=70.0,
            sym_active_df=sym_active_df,
            enable_atr_trailing_prod=True,
            enable_multilayer_rsr=True,
            enable_atr_risk_sizing=True,
            enable_mtf_filter=True,
            risk_sizing_pct=risk_pct,
        )
        row = {
            "risk_pct": risk_pct,
            "cagr": res.get("cagr"), "sharpe": res.get("sharpe"), "max_dd": res.get("max_dd"),
            "calmar": res.get("calmar"), "profit_factor": res.get("profit_factor"),
            "n_trades": res.get("n_trades"), "avg_exposure": res.get("avg_exposure"),
            "win_rate": res.get("win_rate"), "avg_simultaneous_holdings": res.get("avg_simultaneous_holdings"),
        }
        rows.append(row)
        print(f"  risk_pct={risk_pct*100:5.2f}%  CAGR={row['cagr']:+6.1f}%  Sharpe={row['sharpe']:.3f}  "
              f"MaxDD={row['max_dd']:+6.1f}%  Calmar={row['calmar']:.3f}  PF={row['profit_factor']:.3f}  "
              f"Trades={row['n_trades']}  avgExp={row['avg_exposure']:.1f}%")

    os.makedirs("C:/ai-trading/backtests", exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\n結果保存: {OUTPUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
