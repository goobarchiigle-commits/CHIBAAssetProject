"""
backtest/study35a_22pct_reproduction_audit_202606.py
Study35A: 22.4% Reproduction Audit。推測禁止・差分実測のみ。新機能評価は対象外。
一回限りの監査スクリプト（恒久モジュール化しない）。

目的: min_hold_sensitivity_2026-03-31.json の hold3d.IS（CAGR22.4%/Sharpe1.582/MaxDD-12.3%）を
再現できない原因を、実測の差分実験（A/B backtest）で優先度順に特定する。推定ではなく実測。

手法: min_hold_sensitivity.py の run_one() と全く同じ呼び出し（scenario=BASELINE, min_hold=3,
capital=cfg.portfolio.capital, 2018-2024, 静的RSR42, sym_active_df=None）を「現在の再現試行」とし、
記録値とのギャップを確定。その上で、コード調査で判明した「sym_active_df非依存で常時適用される
cfg駆動の新機能候補」を一つずつ無効化し、実測差分を計測する。

候補（コード調査で確認済み・2026-03-31時点に存在しなかった可能性が高いもの）:
  1. gross_exposure_enabled（composite_alpha_bt.py:768, strategy.yaml「2026-04-07追加」の
     gross_exposure_enabled=True/cap値、sym_active_df非依存で常時適用）
  2. rsr_exit閾値（cfg.fujiko.rsr_exit、現在値70.0は「WF5fold検証済2026-06-05」の値）
  3. REENTRY_COOL（TIME_STOP後5日再エントリー禁止、モジュール定数、変更日コメントなし）
  4. CAPITAL参照消失（_bt.CAPITAL→cfg.portfolio.capital、値は同一3,000,000のはずを実測確認）

実行: python src/backtest/study35a_22pct_reproduction_audit_202606.py
"""
from __future__ import annotations
import os, sys, json, warnings, time, dataclasses
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import src.backtest.composite_alpha_bt as bt
from src.config_loader import load_strategy_config
from src.backtest.universe_builder import download_universe

IS_START, IS_END = "2018-01-01", "2024-12-31"
RECORDED = {"cagr": 22.4, "sharpe": 1.582, "max_dd": -12.32}  # min_hold_sensitivity_2026-03-31.json hold3d.IS実測値
OUTPUT_JSON = f"C:/ai-trading/backtests/study35a_22pct_reproduction_audit_202606_{time.strftime('%Y-%m-%d')}.json"


def load_data():
    print("[1/3] ユニバース・価格データ読み込み中（RSR42, 2018-2024）...")
    rsr_syms = bt._load_rsr_universe()
    trade_syms = rsr_syms
    universe_raw = download_universe({**rsr_syms}, start=IS_START, end=IS_END, verbose=False)
    print(f"  取得完了: {len(universe_raw)}銘柄")

    print("[2/3] RSR計算中...")
    rsr_prices = {s: universe_raw[s]["df"]["Close"] for s in rsr_syms if s in universe_raw}
    rsr_df = bt.calc_universe_rsr(rsr_prices)

    print("[3/3] Composite Alpha計算中...")
    alpha_df = bt.calc_composite_alpha_matrix(rsr_prices, window=bt.COMP_ALPHA_WINDOW)
    alpha_df = alpha_df.shift(1)

    # min_hold_sensitivity.py は regime_df を渡していない → run_scenario内部で regime_df 必須箇所のみ要確認
    # （use_regime=False の BASELINE では regime_df は未参照のため、ダミーで代用可）
    import pandas as pd
    regime_df = pd.DataFrame(index=rsr_df.index)

    return universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms


def run(cfg, universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms, *,
        capital=None, rsr_exit_threshold=None):
    return bt.run_scenario(
        scenario="BASELINE",
        universe_raw=universe_raw, rsr_df=rsr_df, alpha_df=alpha_df, regime_df=regime_df,
        trade_syms=trade_syms, rsr_syms=rsr_syms, cfg=cfg,
        start=IS_START, end=IS_END, verbose=False,
        capital=(capital if capital is not None else cfg.portfolio.capital),
        min_hold=3,
        rsr_exit_threshold=rsr_exit_threshold,
    )


def m(res):
    return {"cagr": res.get("cagr"), "sharpe": res.get("sharpe"), "max_dd": res.get("max_dd"), "n_trades": res.get("n_trades")}


def delta(a, b):
    return {k: round(b[k] - a[k], 3) if (isinstance(a.get(k), (int, float)) and isinstance(b.get(k), (int, float))) else None
            for k in ("cagr", "sharpe", "max_dd", "n_trades")}


def main():
    cfg = load_strategy_config()
    universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms = load_data()

    print(f"\n{'='*78}\n  [記録値] min_hold_sensitivity_2026-03-31.json hold3d.IS\n{'='*78}")
    print(f"  CAGR={RECORDED['cagr']:+.1f}%  Sharpe={RECORDED['sharpe']:.3f}  MaxDD={RECORDED['max_dd']:+.1f}%")

    # ── [0] 現在の再現試行（min_hold_sensitivity.py run_one()と完全同一呼び出し）──
    print(f"\n{'='*78}\n  [0] 現在の再現試行（current cfg、デフォルト全使用）\n{'='*78}")
    res_current = run(cfg, universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms)
    cur = m(res_current)
    print(f"  CAGR={cur['cagr']:+.1f}%  Sharpe={cur['sharpe']:.3f}  MaxDD={cur['max_dd']:+.1f}%  trades={cur['n_trades']}")
    gap_total = delta(RECORDED, cur)
    print(f"  記録値との総ギャップ(現在-記録): ΔCAGR={gap_total['cagr']:+.2f}pp  ΔSharpe={gap_total['sharpe']:+.3f}  ΔMaxDD={gap_total['max_dd']:+.2f}pp")

    findings = []

    # ── [1] gross_exposure_enabled OFF（2026-04-07追加、sym_active_df非依存で常時適用）──
    print(f"\n{'='*78}\n  [1] gross_exposure_enabled OFF（cfg.risk_controls、2026-04-07追加分を無効化）\n{'='*78}")
    cfg_ge_off = dataclasses.replace(cfg, risk_controls=dataclasses.replace(cfg.risk_controls, gross_exposure_enabled=False))
    res_ge_off = run(cfg_ge_off, universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms)
    ge_off = m(res_ge_off)
    d_ge = delta(cur, ge_off)
    print(f"  CAGR={ge_off['cagr']:+.1f}%  Sharpe={ge_off['sharpe']:.3f}  MaxDD={ge_off['max_dd']:+.1f}%  trades={ge_off['n_trades']}")
    print(f"  実測差分(OFF-現在): ΔCAGR={d_ge['cagr']:+.2f}pp  ΔSharpe={d_ge['sharpe']:+.3f}  ΔMaxDD={d_ge['max_dd']:+.2f}pp")
    findings.append({"factor": "gross_exposure_enabled(2026-04-07追加)", "measured_delta": d_ge,
                      "explains_pct_of_gap_cagr": round(d_ge["cagr"] / gap_total["cagr"] * 100, 1) if gap_total["cagr"] else None})

    # ── [2] rsr_exit_threshold 70.0(現在) vs 75.0(代替候補) ──
    print(f"\n{'='*78}\n  [2] rsr_exit_threshold: 70.0(現在cfg) vs 75.0(代替候補)\n{'='*78}")
    res_rsr75 = run(cfg, universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms, rsr_exit_threshold=75.0)
    rsr75 = m(res_rsr75)
    d_rsr = delta(cur, rsr75)
    print(f"  CAGR={rsr75['cagr']:+.1f}%  Sharpe={rsr75['sharpe']:.3f}  MaxDD={rsr75['max_dd']:+.1f}%  trades={rsr75['n_trades']}")
    print(f"  実測差分(75.0-現在70.0): ΔCAGR={d_rsr['cagr']:+.2f}pp  ΔSharpe={d_rsr['sharpe']:+.3f}  ΔMaxDD={d_rsr['max_dd']:+.2f}pp")
    findings.append({"factor": "rsr_exit_threshold(70.0→75.0、当時値不明のため両方実測)", "measured_delta": d_rsr,
                      "explains_pct_of_gap_cagr": round(d_rsr["cagr"] / gap_total["cagr"] * 100, 1) if gap_total["cagr"] else None})

    # ── [3] REENTRY_COOL 5(現在) vs 0(無効化候補) ──
    print(f"\n{'='*78}\n  [3] REENTRY_COOL: 5日(現在モジュール定数) vs 0日(無効化候補)\n{'='*78}")
    orig_cool = bt.REENTRY_COOL
    bt.REENTRY_COOL = 0
    res_cool0 = run(cfg, universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms)
    bt.REENTRY_COOL = orig_cool
    cool0 = m(res_cool0)
    d_cool = delta(cur, cool0)
    print(f"  CAGR={cool0['cagr']:+.1f}%  Sharpe={cool0['sharpe']:.3f}  MaxDD={cool0['max_dd']:+.1f}%  trades={cool0['n_trades']}")
    print(f"  実測差分(0日-現在5日): ΔCAGR={d_cool['cagr']:+.2f}pp  ΔSharpe={d_cool['sharpe']:+.3f}  ΔMaxDD={d_cool['max_dd']:+.2f}pp")
    findings.append({"factor": "REENTRY_COOL(5日→0日、変更日コメントなし)", "measured_delta": d_cool,
                      "explains_pct_of_gap_cagr": round(d_cool["cagr"] / gap_total["cagr"] * 100, 1) if gap_total["cagr"] else None})

    # ── [4] CAPITAL参照消失: 明示的3,000,000指定 vs cfg.portfolio.capital経由（値同一性の実測確認）──
    print(f"\n{'='*78}\n  [4] CAPITAL参照消失: capital=3,000,000明示 vs cfg.portfolio.capital経由\n{'='*78}")
    res_cap_explicit = run(cfg, universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms, capital=3_000_000)
    cap_explicit = m(res_cap_explicit)
    d_cap = delta(cur, cap_explicit)
    print(f"  CAGR={cap_explicit['cagr']:+.1f}%  Sharpe={cap_explicit['sharpe']:.3f}  MaxDD={cap_explicit['max_dd']:+.1f}%")
    print(f"  実測差分: ΔCAGR={d_cap['cagr']:+.2f}pp（cfg.portfolio.capital現在値={cfg.portfolio.capital:,}）")
    findings.append({"factor": "CAPITAL参照消失(_bt.CAPITAL→cfg.portfolio.capital)", "measured_delta": d_cap,
                      "verdict": "数値同一のため非要因（構造変更のみ、結果に影響なし）" if d_cap["cagr"] == 0 else "値が異なる、要因あり"})

    # ── [5] 複合: [1]+[2]+[3] 同時適用 ──
    print(f"\n{'='*78}\n  [5] 複合適用: gross_exposure OFF + rsr_exit=75.0 + REENTRY_COOL=0\n{'='*78}")
    bt.REENTRY_COOL = 0
    res_combined = run(cfg_ge_off, universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms, rsr_exit_threshold=75.0)
    bt.REENTRY_COOL = orig_cool
    combined = m(res_combined)
    d_combined_vs_current = delta(cur, combined)
    d_combined_vs_recorded = delta(RECORDED, combined)
    print(f"  CAGR={combined['cagr']:+.1f}%  Sharpe={combined['sharpe']:.3f}  MaxDD={combined['max_dd']:+.1f}%  trades={combined['n_trades']}")
    print(f"  複合差分(vs現在再現): ΔCAGR={d_combined_vs_current['cagr']:+.2f}pp  ΔSharpe={d_combined_vs_current['sharpe']:+.3f}")
    print(f"  記録値との残差(複合適用後): ΔCAGR={d_combined_vs_recorded['cagr']:+.2f}pp  ΔSharpe={d_combined_vs_recorded['sharpe']:+.3f}  ΔMaxDD={d_combined_vs_recorded['max_dd']:+.2f}pp")

    # ── 優先度ランキング（|ΔCAGR|の実測絶対値降順）──
    findings_ranked = sorted(
        [f for f in findings if f.get("measured_delta", {}).get("cagr") is not None],
        key=lambda f: -abs(f["measured_delta"]["cagr"])
    )
    print(f"\n{'='*78}\n  [優先度ランキング] |実測ΔCAGR|降順\n{'='*78}")
    for rank, f in enumerate(findings_ranked, 1):
        d = f["measured_delta"]
        print(f"  {rank}. {f['factor']:55s} ΔCAGR={d['cagr']:+6.2f}pp  ΔSharpe={d['sharpe']:+.3f}  ΔMaxDD={d['max_dd']:+6.2f}pp")

    out = {
        "recorded": RECORDED, "current_reproduction": cur, "total_gap_vs_recorded": gap_total,
        "findings": findings, "combined_test": {"result": combined, "delta_vs_current": d_combined_vs_current,
                                                  "residual_vs_recorded": d_combined_vs_recorded},
        "unmeasurable_factors": [
            "Exit優先順位の変更（2026-03-31時点の優先順位コードが存在しないため代替実装による比較不可）",
            "sizing/cash allocation式そのものの変更（sizing_mode='existing'の代替実装が存在しないため比較不可）",
            "min_sepa/min_rsr/mom_period/turtle_entryが2026-03-31当時から不変だったかの確証（変更日コメントなし）",
        ],
    }
    os.makedirs("C:/ai-trading/backtests", exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n結果保存: {OUTPUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
