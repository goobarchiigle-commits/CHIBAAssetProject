"""
backtest/decay_audit_202606.py
2026-04-13レポート（CAGR22.4%/Sharpe1.582）→2026-06-24 PROD_FAITHFUL（CAGR10.3%/Sharpe0.707）の
寄与度分解。推測禁止・コード実測のみ。一回限りの監査スクリプト（恒久モジュール化しない）。

確認済みの差分要因（コード直接確認）:
  1. Universe差分: min_hold_sensitivity.py の run_one() は sym_active_df を渡していない
     → 動的ユニバース(dyn_rsr42_bear_rs0)・セクター/クラスターキャップが共に未適用
     （composite_alpha_bt.py: _enable_conc_caps = sym_active_df is not None）
  2. Shock Exit差分: min_hold_sensitivity.py は market_shock_mode を渡していない
     → run_scenario デフォルト "full_exit" を使用（現行 main() は strategy.yaml の "composite" を使用）
  3. RSR Exit閾値差分: strategy.yaml の rsr_exit=70.0 は "WF5fold検証済 2026-06-05" とコメントあり
     → 2026-03-31時点はrsr_exit未定義だった可能性が高く、フォールバックでmin_rsr=75.0が
       exit閾値として使われていたと推定されるが、当時のstrategy.yaml実体は失われている（gitなし）ため、
       「75.0 vs 70.0」を両方明示的に実行して両者の効果を直接比較する（仮定に依存しない）。
  4-7. PROD_FAITHFUL追加実装4項目（ATRトレーリング/多層RSR/ATRリスクサイジング/MTF）を1つずつ累積追加。

実行: python src/backtest/decay_audit_202606.py
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
PERIODS = {
    "2018-2024": ("2018-01-01", "2024-12-31"),
    "2020-2024": ("2020-01-01", "2024-12-31"),
    "2025":      ("2025-01-01", "2025-12-31"),
    "2018-2026": ("2018-01-01", "2026-06-23"),
}

OUTPUT_JSON = f"C:/ai-trading/backtests/decay_audit_202606_{time.strftime('%Y-%m-%d')}.json"


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

    # 累積ウォーターフォール定義（順序固定、各stepは前stepに1要因を追加）
    waterfall_steps = [
        ("0_APRIL_REPRO",     dict(use_dyn=False, shock="full_exit", rsr_exit=75.0, atr_trail=False, ml=False, sizing=False, mtf=False)),
        ("1_+UNIVERSE_CAPS",  dict(use_dyn=True,  shock="full_exit", rsr_exit=75.0, atr_trail=False, ml=False, sizing=False, mtf=False)),
        ("2_+SHOCK_COMPOSITE",dict(use_dyn=True,  shock="composite", rsr_exit=75.0, atr_trail=False, ml=False, sizing=False, mtf=False)),
        ("3_+RSR_EXIT_70",    dict(use_dyn=True,  shock="composite", rsr_exit=70.0, atr_trail=False, ml=False, sizing=False, mtf=False)),
        ("4_+ATR_TRAILING",   dict(use_dyn=True,  shock="composite", rsr_exit=70.0, atr_trail=True,  ml=False, sizing=False, mtf=False)),
        ("5_+MULTILAYER_RSR", dict(use_dyn=True,  shock="composite", rsr_exit=70.0, atr_trail=True,  ml=True,  sizing=False, mtf=False)),
        ("6_+ATR_RISK_SIZING",dict(use_dyn=True,  shock="composite", rsr_exit=70.0, atr_trail=True,  ml=True,  sizing=True,  mtf=False)),
        ("7_+MTF_FILTER",     dict(use_dyn=True,  shock="composite", rsr_exit=70.0, atr_trail=True,  ml=True,  sizing=True,  mtf=True)),
    ]

    all_results: dict[str, list[dict]] = {}
    for period_label, (p_start, p_end) in PERIODS.items():
        steps_for_period = (
            waterfall_steps if period_label in ("2018-2024", "2020-2024")
            else [waterfall_steps[0], waterfall_steps[3], waterfall_steps[-1]]  # April構成 / 現行BASELINE相当 / PROD_FAITHFUL
        )
        print(f"\n{'='*70}\n  期間: {period_label} ({p_start}〜{p_end})\n{'='*70}")
        rows = []
        for label, params in steps_for_period:
            sym_active = sym_active_df if params["use_dyn"] else None
            res = bt.run_scenario(
                scenario="PROD_FAITHFUL" if (params["atr_trail"] or params["ml"] or params["sizing"] or params["mtf"]) else "BASELINE",
                universe_raw=universe_raw, rsr_df=rsr_df, alpha_df=alpha_df, regime_df=regime_df,
                trade_syms=trade_syms, rsr_syms=rsr_syms, cfg=cfg,
                start=p_start, end=p_end, verbose=False,
                tech_matrices=tech_matrices,
                capital=cfg.portfolio.capital,
                min_hold=cfg.risk.min_hold_days,
                market_shock_mode=params["shock"],
                rsr_exit_threshold=params["rsr_exit"],
                sym_active_df=sym_active,
                enable_atr_trailing_prod=params["atr_trail"],
                enable_multilayer_rsr=params["ml"],
                enable_atr_risk_sizing=params["sizing"],
                enable_mtf_filter=params["mtf"],
            )
            row = {
                "label": label, "cagr": res.get("cagr"), "sharpe": res.get("sharpe"),
                "max_dd": res.get("max_dd"), "calmar": res.get("calmar"),
                "profit_factor": res.get("profit_factor"), "n_trades": res.get("n_trades"),
                "n_trades_yr": res.get("n_trades_yr"), "win_rate": res.get("win_rate"),
                "avg_hold_days": res.get("avg_hold_days"),
                "avg_candidates": res.get("avg_candidates"),
                "avg_simultaneous_holdings": res.get("avg_simultaneous_holdings"),
                "avg_exposure": res.get("avg_exposure"),
                "exit_reason_counts": res.get("exit_reason_counts"),
            }
            rows.append(row)
            print(f"  {label:22s} CAGR={row['cagr']:+6.1f}%  Sharpe={row['sharpe']:.3f}  "
                  f"MaxDD={row['max_dd']:+6.1f}%  trades={row['n_trades']}  avgCand={row['avg_candidates']}")
        all_results[period_label] = rows

    os.makedirs("C:/ai-trading/backtests", exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n結果保存: {OUTPUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
