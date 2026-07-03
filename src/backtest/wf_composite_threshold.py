"""
backtest/wf_composite_threshold.py
Phase 3 Step 3: composite shock exit 閾値最適化 WF検証

目的: 2022/2024 macroDDの改善
現行: market_ret <= -0.05 AND sym_ret <= -0.08
比較: (-4%/-7%), (-5%/-8%), (-6%/-9%)

評価優先: worst OOS DD → WF median Sharpe → 2025 Sharpe

実行:
  cd C:/ai-trading
  python src/backtest/wf_composite_threshold.py
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
from src.utils.memory import collect_and_log_process_memory

os.environ.pop("DATA_VERSION", None)

# ------------------------------------------------------------------ #
# 設定
# ------------------------------------------------------------------ #
# composite モードのパラメータをグリッドサーチ
THRESHOLD_CONFIGS = [
    {"name": "composite_4_7",  "market_thr": -0.04, "sym_thr": -0.07},
    {"name": "composite_5_8",  "market_thr": -0.05, "sym_thr": -0.08},   # 現行
    {"name": "composite_6_9",  "market_thr": -0.06, "sym_thr": -0.09},
]

WF_SEGS = [
    {"seg": 1, "is": ("2018-01-01", "2019-12-31"), "oos": ("2020-01-01", "2020-12-31")},
    {"seg": 2, "is": ("2019-01-01", "2020-12-31"), "oos": ("2021-01-01", "2021-12-31")},
    {"seg": 3, "is": ("2020-01-01", "2021-12-31"), "oos": ("2022-01-01", "2022-12-31")},
    {"seg": 4, "is": ("2021-01-01", "2022-12-31"), "oos": ("2023-01-01", "2023-12-31")},
    {"seg": 5, "is": ("2022-01-01", "2023-12-31"), "oos": ("2024-01-01", "2024-12-31")},
]
TRUE_OOS = ("2025-01-01", "2025-12-31")
DATA_END  = "2025-12-31"
IS_FULL   = ("2018-01-01", "2024-12-31")

OUTPUT_DIR  = "C:/ai-trading/backtests"
OUTPUT_JSON = os.path.join(OUTPUT_DIR, f"wf_composite_threshold_{time.strftime('%Y-%m-%d')}.json")


# ------------------------------------------------------------------ #
# composite 閾値を動的に注入するパッチ
# ------------------------------------------------------------------ #
def _patch_composite_thresholds(market_thr: float, sym_thr: float):
    """
    run_scenario() の composite モード判定閾値をモンキーパッチで変更する。

    composite モードのコードは composite_alpha_bt.py 内の以下の部分:
      if market_shock and market_shock_mode == "composite":
          if sym_ret_i <= -0.08:  ← ここを変える

    パッチ手法: _bt モジュールの定数を一時的に上書きするのではなく、
    _bt.run_scenario の wrapperを使い、market_shock_mode に
    カスタム閾値を埋め込んだ新しいモードを渡す。

    実装: composite_alpha_bt.py に composite_market_thr / composite_sym_thr
    パラメータを追加するのが理想だが、この WF スクリプトでは
    monkey-patch で対応する（プロダクションコード変更なし）。
    """
    import backtest.composite_alpha_bt as _mod
    _mod._COMPOSITE_MARKET_THR = market_thr
    _mod._COMPOSITE_SYM_THR    = sym_thr


# composite_alpha_bt.py の内部ループで閾値を読む変数を使っているか確認。
# 現行コードは -0.05 / -0.08 がハードコード。
# → composite_alpha_bt.py にパラメータを追加して対応する。

def load_data(cfg):
    print("[1/2] データロード中（RSR42, 2018-2025）...")
    rsr_syms     = _bt._load_rsr_universe()
    universe_raw = download_universe({**rsr_syms}, start="2018-01-01", end=DATA_END, verbose=False)
    topix_close  = _bt._download_topix("2018-01-01", DATA_END)

    print("[2/2] RSR・Alpha計算中...")
    rsr42_prices = {s: universe_raw[s]["df"]["Close"] for s in rsr_syms if s in universe_raw}
    rsr_df = calc_universe_rsr(rsr42_prices)
    trade_prices = {s: universe_raw[s]["df"]["Close"] for s in rsr_syms if s in universe_raw}
    alpha_df = _bt.calc_composite_alpha_matrix(trade_prices, window=_bt.COMP_ALPHA_WINDOW)
    alpha_df = alpha_df.shift(1)
    regime_df     = _bt._calc_regime(topix_close)
    tech_matrices = _bt._precompute_tech_matrices(universe_raw, list(rsr_syms.keys()))
    return (universe_raw, rsr_df, alpha_df, regime_df, rsr_syms, tech_matrices, topix_close, cfg)


def run_seg(universe_raw, rsr_df, alpha_df, regime_df, rsr_syms,
            tech_matrices, topix_close, cfg,
            start: str, end: str,
            market_thr: float, sym_thr: float) -> dict:
    return _bt.run_scenario(
        scenario         = "BASELINE",
        universe_raw     = universe_raw,
        rsr_df           = rsr_df,
        alpha_df         = alpha_df,
        regime_df        = regime_df,
        trade_syms       = rsr_syms,
        rsr_syms         = rsr_syms,
        cfg              = cfg,
        start            = start,
        end              = end,
        capital          = float(cfg.portfolio.capital),
        verbose          = False,
        tech_matrices    = tech_matrices,
        topix_close      = topix_close,
        min_hold         = int(cfg.risk.min_hold_days),
        market_shock_mode    = "composite",
        composite_market_thr = market_thr,
        composite_sym_thr    = sym_thr,
        enable_volume_filter     = True,
        enable_volatility_filter = False,
        enable_atr_filter        = False,
        enable_market_filter     = False,
    )


def run_wf_for_threshold(thr_cfg, universe_raw, rsr_df, alpha_df, regime_df,
                          rsr_syms, tech_matrices, topix_close, cfg):
    name = thr_cfg["name"]
    mt   = thr_cfg["market_thr"]
    st   = thr_cfg["sym_thr"]
    seg_results = []

    for seg_def in WF_SEGS:
        n = seg_def["seg"]
        is_s, is_e = seg_def["is"]
        oos_s, oos_e = seg_def["oos"]
        r_is  = run_seg(universe_raw, rsr_df, alpha_df, regime_df, rsr_syms,
                        tech_matrices, topix_close, cfg, is_s, is_e, mt, st)
        r_oos = run_seg(universe_raw, rsr_df, alpha_df, regime_df, rsr_syms,
                        tech_matrices, topix_close, cfg, oos_s, oos_e, mt, st)
        is_sh  = r_is.get("sharpe",  0) if r_is  else 0.0
        oos_sh = r_oos.get("sharpe", 0) if r_oos else 0.0
        oos_dd = r_oos.get("max_dd", 0) if r_oos else 0.0
        ratio  = round(oos_sh / is_sh, 3) if is_sh != 0 else None
        win    = oos_sh > 0
        seg_results.append({
            "seg": n, "oos": f"{oos_s[:4]}",
            "is_sharpe": round(is_sh, 3), "oos_sharpe": round(oos_sh, 3),
            "oos_max_dd": round(oos_dd, 2), "ratio": ratio, "win": win,
        })
        del r_is, r_oos

    r_is_full = run_seg(universe_raw, rsr_df, alpha_df, regime_df, rsr_syms,
                        tech_matrices, topix_close, cfg, *IS_FULL, mt, st)
    r_oos25   = run_seg(universe_raw, rsr_df, alpha_df, regime_df, rsr_syms,
                        tech_matrices, topix_close, cfg, *TRUE_OOS, mt, st)

    pass_count = sum(1 for s in seg_results if s["win"])
    oos_sharpes = [s["oos_sharpe"] for s in seg_results]
    median_oos  = float(np.median(oos_sharpes))
    worst_dd    = min(s["oos_max_dd"] for s in seg_results)
    ratios      = [s["ratio"] for s in seg_results if s["ratio"] is not None]
    avg_ratio   = float(np.mean(ratios)) if ratios else 0.0

    collect_and_log_process_memory(f"thr_{name}")
    return {
        "config_name": name,
        "market_thr":  mt,
        "sym_thr":     st,
        "segments":    seg_results,
        "wf_summary": {
            "pass_count":        f"{pass_count}/5",
            "median_oos_sharpe": round(median_oos, 3),
            "worst_oos_dd":      round(worst_dd, 2),
            "avg_oos_is_ratio":  round(avg_ratio, 3),
        },
        "full_is":       {"sharpe": round(r_is_full.get("sharpe", 0), 3) if r_is_full else 0,
                          "max_dd": round(r_is_full.get("max_dd",  0), 2) if r_is_full else 0},
        "true_oos_2025": {"sharpe": round(r_oos25.get("sharpe", 0), 3) if r_oos25 else 0,
                          "max_dd": round(r_oos25.get("max_dd",  0), 2) if r_oos25 else 0},
    }


def main():
    cfg = load_strategy_config()
    print("=" * 72)
    print("  Phase 3 Step 3 WF: composite shock exit 閾値最適化")
    print("  目的: 2022/2024 macro DD 改善  評価優先: worst DD > Sharpe")
    print("=" * 72 + "\n")

    data = load_data(cfg)
    universe_raw, rsr_df, alpha_df, regime_df, rsr_syms, tech_matrices, topix_close, cfg = data

    all_results = []
    for thr_cfg in THRESHOLD_CONFIGS:
        name = thr_cfg["name"]
        print(f"\n{'─'*60}")
        print(f"  設定: {name}  market<={thr_cfg['market_thr']:.0%}  sym<={thr_cfg['sym_thr']:.0%}")
        result = run_wf_for_threshold(thr_cfg, *data)
        all_results.append(result)
        for seg in result["segments"]:
            mark = "OK" if seg["win"] else "NG"
            print(f"    Seg{seg['seg']} OOS={seg['oos']}  IS={seg['is_sharpe']:.3f}"
                  f"  OOS={seg['oos_sharpe']:.3f}  DD={seg['oos_max_dd']:+.1f}%  {mark}")
        s = result["wf_summary"]
        fi = result["full_is"]
        t  = result["true_oos_2025"]
        print(f"  WF: {s['pass_count']} / median={s['median_oos_sharpe']:.3f} / worstDD={s['worst_oos_dd']:+.1f}%")
        print(f"  IS Sharpe={fi['sharpe']:.3f}  MaxDD={fi['max_dd']:+.1f}%")
        print(f"  2025 Sharpe={t['sharpe']:.3f}  MaxDD={t['max_dd']:+.1f}%")

    print("\n" + "=" * 72)
    print("  比較サマリー（worst DD改善優先）")
    print(f"  {'設定':<20}  {'WF勝率':>6}  {'中央値':>7}  {'worstDD':>9}  {'IS Sh':>6}  {'2025 Sh':>7}")
    print("  " + "-" * 60)
    base_dd = all_results[1]["wf_summary"]["worst_oos_dd"]   # 現行 -5/-8
    for r in all_results:
        s = r["wf_summary"]
        fi = r["full_is"]
        t  = r["true_oos_2025"]
        mark = "★" if s["worst_oos_dd"] > base_dd + 0.3 else ("現行" if r["config_name"] == "composite_5_8" else " ")
        print(f"  {mark}{r['config_name']:<19}  {s['pass_count']:>6}  "
              f"{s['median_oos_sharpe']:>7.3f}  {s['worst_oos_dd']:>+9.1f}%"
              f"  {fi['sharpe']:>6.3f}  {t['sharpe']:>7.3f}")
    print("=" * 72)

    output = {
        "date":        time.strftime("%Y-%m-%d"),
        "description": "Phase3 Step3: composite閾値最適化",
        "results":     all_results,
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  結果保存: {OUTPUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
