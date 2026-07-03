"""
backtest/post_atr_removal_validation_202606.py
Study33: Post-ATR Removal Validation。推測禁止・コード実測のみ。
一回限りの監査スクリプト（恒久モジュール化しない）。

目的: ATR Risk Sizing除去（Study32 decision=REMOVE, confidence=HIGH）後の真のベースラインを確定。

比較:
  A. APRIL_REPRO        : sizing_mode="existing"（旧来デフォルト）, PROD_FAITHFUL系機能すべてOFF
  B. CURRENT_PROD        : PROD_FAITHFUL全機能ON（ATR Risk Sizing含む, risk_pct=1.25%）
  C. CURRENT_PROD - ATR  : PROD_FAITHFUL機能ON, ATR Risk Sizingのみ除去 → sizing_mode="equal"
                           （Study32のREMOVE判定後の代替方式として採用、A_EQUALと同一設定）

寄与度分解（APRIL_REPRO → PROD minus ATR、sizing_modeはAPRIL_REPRO側"existing"で固定し
4要因のみを単独追加。Universe = 動的ユニバース+市場ショックcomposite化+RSR Exit閾値70への変更を
一括（本ラウンドで新規追加された機能ではなく既存ロック済み変更のため一括計上）。
最後にsizing_mode "existing"→"equal" への切替を分解外の補正行として明示（CのEqual Weightへの
切替はATR除去の代替方式選定であり、4要因のいずれにも属さない）。

実行: python src/backtest/post_atr_removal_validation_202606.py
"""
from __future__ import annotations
import os, sys, json, warnings, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import src.backtest.composite_alpha_bt as bt
from src.backtest.atr_sizing_diagnostic_202606 import load_all

FULL_START = "2018-01-01"
FULL_END   = "2026-06-23"
OUTPUT_JSON = f"C:/ai-trading/backtests/post_atr_removal_validation_202606_{time.strftime('%Y-%m-%d')}.json"

METRIC_KEYS = ("cagr", "sharpe", "max_dd", "calmar", "profit_factor", "avg_exposure", "n_trades")


def run(cfg, universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms, tech_matrices,
        sym_active_df, *, use_dyn, shock, rsr_exit, atr_trail, ml, sizing, mtf, sizing_mode):
    return bt.run_scenario(
        scenario="PROD_FAITHFUL" if (atr_trail or ml or sizing or mtf) else "BASELINE",
        universe_raw=universe_raw, rsr_df=rsr_df, alpha_df=alpha_df, regime_df=regime_df,
        trade_syms=trade_syms, rsr_syms=rsr_syms, cfg=cfg,
        start=FULL_START, end=FULL_END, verbose=False,
        tech_matrices=tech_matrices,
        capital=cfg.portfolio.capital,
        min_hold=cfg.risk.min_hold_days,
        market_shock_mode=shock,
        rsr_exit_threshold=rsr_exit,
        sym_active_df=(sym_active_df if use_dyn else None),
        enable_atr_trailing_prod=atr_trail,
        enable_multilayer_rsr=ml,
        enable_atr_risk_sizing=sizing,
        enable_mtf_filter=mtf,
        risk_sizing_pct=bt.PROD_RISK_PCT,
        sizing_mode=sizing_mode,
    )


def row_of(label, res):
    return {"label": label, **{k: res.get(k) for k in METRIC_KEYS}}


def print_row(r):
    print(f"  {r['label']:22s} CAGR={r['cagr']:+6.1f}%  Sharpe={r['sharpe']:.3f}  MaxDD={r['max_dd']:+6.1f}%  "
          f"Calmar={r['calmar']:.3f}  PF={r['profit_factor']:.3f}  AvgExp={r['avg_exposure']:.1f}%  trades={r['n_trades']}")


def main():
    cfg, universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms, sym_active_df, tech_matrices = load_all()
    args = (cfg, universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms, tech_matrices, sym_active_df)

    print(f"\n{'='*78}\n  [A/B/C] ヘッドライン比較（期間: {FULL_START}〜{FULL_END}）\n{'='*78}")

    res_a = run(*args, use_dyn=False, shock="full_exit", rsr_exit=75.0,
                atr_trail=False, ml=False, sizing=False, mtf=False, sizing_mode="existing")
    res_b = run(*args, use_dyn=True, shock="composite", rsr_exit=70.0,
                atr_trail=True, ml=True, sizing=True, mtf=True, sizing_mode="existing")
    res_c = run(*args, use_dyn=True, shock="composite", rsr_exit=70.0,
                atr_trail=True, ml=True, sizing=False, mtf=True, sizing_mode="equal")

    row_a, row_b, row_c = row_of("A_APRIL_REPRO", res_a), row_of("B_CURRENT_PROD", res_b), row_of("C_PROD_MINUS_ATR", res_c)
    for r in (row_a, row_b, row_c):
        print_row(r)

    # ---------------------------------------------------------------- #
    # 寄与度分解: APRIL_REPRO → PROD minus ATR（sizing_mode="existing"で固定、4要因のみ単独追加）
    # ---------------------------------------------------------------- #
    print(f"\n{'='*78}\n  [寄与度分解] APRIL_REPRO → PROD minus ATR（sizing_mode=existing固定）\n{'='*78}")
    waterfall_steps = [
        ("0_APRIL_REPRO",      dict(use_dyn=False, shock="full_exit", rsr_exit=75.0, atr_trail=False, ml=False, mtf=False)),
        ("1_+UNIVERSE",        dict(use_dyn=True,  shock="composite", rsr_exit=70.0, atr_trail=False, ml=False, mtf=False)),
        ("2_+ATR_TRAILING",    dict(use_dyn=True,  shock="composite", rsr_exit=70.0, atr_trail=True,  ml=False, mtf=False)),
        ("3_+MULTILAYER_RSR",  dict(use_dyn=True,  shock="composite", rsr_exit=70.0, atr_trail=True,  ml=True,  mtf=False)),
        ("4_+MTF_FILTER",      dict(use_dyn=True,  shock="composite", rsr_exit=70.0, atr_trail=True,  ml=True,  mtf=True)),
    ]
    wf_rows = []
    prev_cagr = prev_sharpe = None
    for label, p in waterfall_steps:
        res = run(*args, use_dyn=p["use_dyn"], shock=p["shock"], rsr_exit=p["rsr_exit"],
                  atr_trail=p["atr_trail"], ml=p["ml"], sizing=False, mtf=p["mtf"], sizing_mode="existing")
        row = row_of(label, res)
        d_cagr = (row["cagr"] - prev_cagr) if prev_cagr is not None else None
        d_sharpe = (row["sharpe"] - prev_sharpe) if prev_sharpe is not None else None
        row["delta_cagr_pp"] = round(d_cagr, 2) if d_cagr is not None else None
        row["delta_sharpe"] = round(d_sharpe, 3) if d_sharpe is not None else None
        wf_rows.append(row)
        prev_cagr, prev_sharpe = row["cagr"], row["sharpe"]
        dcagr_s = f"{row['delta_cagr_pp']:+.2f}pp" if row["delta_cagr_pp"] is not None else "  --  "
        dsharpe_s = f"{row['delta_sharpe']:+.3f}" if row["delta_sharpe"] is not None else " -- "
        print(f"  {label:22s} CAGR={row['cagr']:+6.1f}%({dcagr_s})  Sharpe={row['sharpe']:.3f}({dsharpe_s})  "
              f"MaxDD={row['max_dd']:+6.1f}%  Calmar={row['calmar']:.3f}  PF={row['profit_factor']:.3f}  trades={row['n_trades']}")

    # ---------------------------------------------------------------- #
    # 補正行: sizing_mode "existing"→"equal" 切替（4要因の外、ATR除去の代替方式選定）
    # ---------------------------------------------------------------- #
    print(f"\n{'='*78}\n  [補正行] sizing_mode existing→equal 切替（4要因の外、Study32代替方式選定）\n{'='*78}")
    d_cagr_sizing = row_c["cagr"] - wf_rows[-1]["cagr"]
    d_sharpe_sizing = row_c["sharpe"] - wf_rows[-1]["sharpe"]
    print(f"  4_+MTF_FILTER(existing) → C_PROD_MINUS_ATR(equal)  "
          f"ΔCAGR={d_cagr_sizing:+.2f}pp  ΔSharpe={d_sharpe_sizing:+.3f}")

    out = {
        "period": [FULL_START, FULL_END],
        "headline": {"A_APRIL_REPRO": row_a, "B_CURRENT_PROD": row_b, "C_PROD_MINUS_ATR": row_c},
        "waterfall_existing_sizing": wf_rows,
        "sizing_mode_correction": {
            "from": "4_+MTF_FILTER(existing)", "to": "C_PROD_MINUS_ATR(equal)",
            "delta_cagr_pp": round(d_cagr_sizing, 2), "delta_sharpe": round(d_sharpe_sizing, 3),
        },
    }
    os.makedirs("C:/ai-trading/backtests", exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n結果保存: {OUTPUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
