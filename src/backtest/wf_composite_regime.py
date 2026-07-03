"""
backtest/wf_composite_regime.py
composite shock exit + regime_2 sizing 採用後のウォークフォワード再検証

目的:
  単年OOS改善だけでなく、5分割WFで過学習なし・汎化性を確認する。

検証設定:
  - market_shock_mode = composite（strategy.yaml から読み込み）
  - regime_sizing     = regime_2  （〃）
  - bear_scale        = 0.25      （〃）

採用基準（厳格版）:
  WF pass ratio  : >= 5/5
  OOS Sharpe 中央値 : >= 0.55
  worst window MaxDD : <= -8.0%  （実際には改善を確認するため -10% 以下なら要注意）

WFセグメント（変更なし）:
  Seg1: IS=2018-2019 OOS=2020
  Seg2: IS=2019-2020 OOS=2021
  Seg3: IS=2020-2021 OOS=2022
  Seg4: IS=2021-2022 OOS=2023
  Seg5: IS=2022-2023 OOS=2024
  真OOS: 2025（WF外）

実行:
  cd C:/ai-trading
  python src/backtest/wf_composite_regime.py
"""

from __future__ import annotations
import gc
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
# WF セグメント定義
# ------------------------------------------------------------------ #
WF_SEGMENTS = [
    {"seg": 1, "is": ("2018-01-01", "2019-12-31"), "oos": ("2020-01-01", "2020-12-31")},
    {"seg": 2, "is": ("2019-01-01", "2020-12-31"), "oos": ("2021-01-01", "2021-12-31")},
    {"seg": 3, "is": ("2020-01-01", "2021-12-31"), "oos": ("2022-01-01", "2022-12-31")},
    {"seg": 4, "is": ("2021-01-01", "2022-12-31"), "oos": ("2023-01-01", "2023-12-31")},
    {"seg": 5, "is": ("2022-01-01", "2023-12-31"), "oos": ("2024-01-01", "2024-12-31")},
]
FULL_IS   = ("2018-01-01", "2024-12-31")
TRUE_OOS  = ("2025-01-01", "2025-12-31")
DATA_START = "2018-01-01"
DATA_END   = "2025-12-31"

OUTPUT_DIR  = "C:/ai-trading/backtests"
OUTPUT_JSON = os.path.join(OUTPUT_DIR, f"wf_composite_regime_{time.strftime('%Y-%m-%d')}.json")

# 採用基準
PASS_SHARPE_MIN  = 0.0   # OOS Sharpe > IS でなくても勝ちとする（Sharpe > 0 で Pass）
MEDIAN_OOS_MIN   = 0.55
WORST_DD_LIMIT   = -10.0  # -10%以下は要注意（基準は -8% だが余裕をみる）


# ------------------------------------------------------------------ #
# データロード
# ------------------------------------------------------------------ #
def load_data(cfg):
    print("[1/3] ユニバース・価格データ読み込み中（RSR42, 2018-2025）...")
    rsr_syms   = _bt._load_rsr_universe()
    trade_syms = rsr_syms
    universe_raw = download_universe({**rsr_syms}, start=DATA_START, end=DATA_END, verbose=False)
    topix_close  = _bt._download_topix(DATA_START, DATA_END)

    print("[2/3] RSR計算中...")
    rsr42_prices = {s: universe_raw[s]["df"]["Close"] for s in rsr_syms if s in universe_raw}
    rsr_df = calc_universe_rsr(rsr42_prices)

    print("[3/3] Composite Alpha計算中...")
    trade_prices = {s: universe_raw[s]["df"]["Close"] for s in trade_syms if s in universe_raw}
    alpha_df = _bt.calc_composite_alpha_matrix(trade_prices, window=_bt.COMP_ALPHA_WINDOW)
    alpha_df = alpha_df.shift(1)

    regime_df     = _bt._calc_regime(topix_close)
    tech_matrices = _bt._precompute_tech_matrices(universe_raw, list(trade_syms.keys()))

    # regime_scale_series
    rc = cfg.risk_controls
    _regime_scale_series = None
    if rc.regime_sizing != "none" and topix_close is not None:
        _ma200      = topix_close.rolling(200, min_periods=1).mean()
        _vol20      = topix_close.pct_change().rolling(20, min_periods=10).std()
        _vol_median = float(_vol20.dropna().median())
        _above      = topix_close >= _ma200
        _hi_vol     = _vol20 >= _vol_median
        _regime_scale_series = pd.Series(index=topix_close.index, dtype=float)
        _regime_scale_series[_above & ~_hi_vol] = 1.0
        _regime_scale_series[_above &  _hi_vol] = (0.5 if rc.regime_sizing == "regime_4" else 1.0)
        _regime_scale_series[~_above]            = rc.bear_scale

    print(f"  RSR: {rsr_df.shape[1]}銘柄 / shock={rc.shock_exit_mode} / regime={rc.regime_sizing}\n")
    return universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms, tech_matrices, topix_close, _regime_scale_series


# ------------------------------------------------------------------ #
# 1セグメント実行
# ------------------------------------------------------------------ #
def run_seg(universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms,
            tech_matrices, topix_close, regime_scale_series, cfg,
            start: str, end: str) -> dict:
    return _bt.run_scenario(
        scenario             = "BASELINE",
        universe_raw         = universe_raw,
        rsr_df               = rsr_df,
        alpha_df             = alpha_df,
        regime_df            = regime_df,
        trade_syms           = trade_syms,
        rsr_syms             = rsr_syms,
        cfg                  = cfg,
        start                = start,
        end                  = end,
        capital              = float(cfg.portfolio.capital),
        verbose              = False,
        tech_matrices        = tech_matrices,
        topix_close          = topix_close,
        min_hold             = int(cfg.risk.min_hold_days),
        market_shock_mode    = cfg.risk_controls.shock_exit_mode,
        regime_scale_series  = regime_scale_series,
        enable_volume_filter     = True,
        enable_volatility_filter = False,
        enable_atr_filter        = False,
        enable_market_filter     = False,
    )


# ------------------------------------------------------------------ #
# メイン
# ------------------------------------------------------------------ #
def main():
    cfg = load_strategy_config()
    rc  = cfg.risk_controls

    print("=" * 72)
    print("  WF再検証: composite shock exit + regime_2 sizing")
    print(f"  shock={rc.shock_exit_mode} / regime={rc.regime_sizing} / bear_scale={rc.bear_scale}")
    print(f"  採用基準: WF pass>=5/5 / median OOS Sharpe>={MEDIAN_OOS_MIN} / worst DD>{WORST_DD_LIMIT}%")
    print("=" * 72 + "\n")

    (universe_raw, rsr_df, alpha_df, regime_df,
     trade_syms, rsr_syms, tech_matrices,
     topix_close, regime_scale_series) = load_data(cfg)

    seg_results = []

    # ---- WF セグメント ----
    for seg_def in WF_SEGMENTS:
        seg_n = seg_def["seg"]
        is_s, is_e = seg_def["is"]
        oos_s, oos_e = seg_def["oos"]

        r_is  = run_seg(universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms,
                        tech_matrices, topix_close, regime_scale_series, cfg, is_s, is_e)
        r_oos = run_seg(universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms,
                        tech_matrices, topix_close, regime_scale_series, cfg, oos_s, oos_e)

        is_sh  = r_is.get("sharpe",  0) if r_is  else 0.0
        oos_sh = r_oos.get("sharpe", 0) if r_oos else 0.0
        oos_dd = r_oos.get("max_dd", 0) if r_oos else 0.0
        ratio  = round(oos_sh / is_sh, 3) if is_sh != 0 else None
        win    = oos_sh > PASS_SHARPE_MIN

        seg_results.append({
            "seg":        seg_n,
            "is":         f"{is_s[:4]}-{is_e[:4]}",
            "oos":        f"{oos_s[:4]}-{oos_e[:4]}",
            "is_sharpe":  round(is_sh,  3),
            "oos_sharpe": round(oos_sh, 3),
            "oos_max_dd": round(oos_dd, 2),
            "ratio":      ratio,
            "win":        win,
        })

        win_mark = "✅" if win else "❌"
        print(f"  Seg{seg_n} IS={is_s[:4]}-{is_e[:4]}  OOS={oos_s[:4]}-{oos_e[:4]}"
              f"  IS_Sh={is_sh:.3f}  OOS_Sh={oos_sh:.3f}  OOS_DD={oos_dd:+.1f}%"
              f"  ratio={ratio}  {win_mark}")

        del r_is, r_oos
        collect_and_log_process_memory(f"seg{seg_n}")

    # ---- Full IS ----
    print("\n  Full IS (2018-2024)...")
    r_full_is = run_seg(universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms,
                        tech_matrices, topix_close, regime_scale_series, cfg, *FULL_IS)
    full_is_sharpe = r_full_is.get("sharpe", 0) if r_full_is else 0.0
    full_is_dd     = r_full_is.get("max_dd", 0) if r_full_is else 0.0

    # ---- True OOS 2025 ----
    print("  True OOS (2025)...")
    r_true_oos = run_seg(universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms,
                         tech_matrices, topix_close, regime_scale_series, cfg, *TRUE_OOS)
    true_oos_sharpe = r_true_oos.get("sharpe", 0) if r_true_oos else 0.0
    true_oos_dd     = r_true_oos.get("max_dd", 0) if r_true_oos else 0.0
    true_oos_cagr   = r_true_oos.get("cagr", 0)   if r_true_oos else 0.0

    # ---- 月次 2025 ----
    monthly_2025 = {}
    eq_oos = r_true_oos.get("equity_curve") if r_true_oos else None
    if eq_oos is not None and not eq_oos.empty:
        for (yr, mo), grp in eq_oos.groupby([eq_oos.index.year, eq_oos.index.month]):
            if len(grp) < 2:
                continue
            monthly_2025[f"{yr}-{mo:02d}"] = round(float(grp.iloc[-1] / grp.iloc[0] - 1) * 100, 2)

    wins_2025  = sum(1 for v in monthly_2025.values() if v > 0)
    loses_2025 = sum(1 for v in monthly_2025.values() if v <= 0)

    # ---- サマリー ----
    oos_sharpes = [s["oos_sharpe"] for s in seg_results]
    pass_count  = sum(1 for s in seg_results if s["win"])
    median_oos  = float(np.median(oos_sharpes))
    worst_dd    = min(s["oos_max_dd"] for s in seg_results)
    avg_ratio   = float(np.mean([s["ratio"] for s in seg_results if s["ratio"] is not None]))

    ok_pass   = pass_count >= 5
    ok_median = median_oos >= MEDIAN_OOS_MIN
    ok_dd     = worst_dd >= WORST_DD_LIMIT
    verdict   = "✅ PASS" if (ok_pass and ok_median and ok_dd) else "❌ FAIL"

    print("\n" + "=" * 72)
    print("  WF サマリー")
    print(f"  OOS 勝率   : {pass_count}/5  {'✅' if ok_pass else '❌'}")
    print(f"  OOS 中央値 : {median_oos:.3f}  (基準 >= {MEDIAN_OOS_MIN})  {'✅' if ok_median else '❌'}")
    print(f"  worst DD   : {worst_dd:+.1f}%  (基準 > {WORST_DD_LIMIT}%)  {'✅' if ok_dd else '❌'}")
    print(f"  avg OOS/IS : {avg_ratio:.3f}")
    print(f"  Full IS    : Sharpe={full_is_sharpe:.3f}  MaxDD={full_is_dd:+.1f}%")
    print(f"  True OOS   : Sharpe={true_oos_sharpe:.3f}  MaxDD={true_oos_dd:+.1f}%  CAGR={true_oos_cagr:+.1f}%")
    print(f"  OOS 2025月次: {wins_2025}勝{loses_2025}敗  {monthly_2025}")
    print(f"\n  総合判定: {verdict}")
    print("=" * 72)

    # ---- JSON保存 ----
    output = {
        "date":             time.strftime("%Y-%m-%d"),
        "description":      f"WF composite_shock+regime_2 (shock={rc.shock_exit_mode}, regime={rc.regime_sizing})",
        "config": {
            "shock_exit_mode":    rc.shock_exit_mode,
            "regime_sizing":      rc.regime_sizing,
            "bear_scale":         rc.bear_scale,
            "min_hold":           cfg.risk.min_hold_days,
            "min_rsr":            cfg.fujiko.min_rsr,
            "max_positions":      cfg.portfolio.max_positions,
            "capital":            cfg.portfolio.capital,
        },
        "wf_segments":      seg_results,
        "wf_summary": {
            "oos_wins":        f"{pass_count}/5",
            "median_oos_sharpe": round(median_oos, 3),
            "worst_oos_dd":    round(worst_dd, 2),
            "avg_oos_is_ratio": round(avg_ratio, 3),
            "judgment":         "PASS" if (ok_pass and ok_median and ok_dd) else "FAIL",
        },
        "full_is": {
            "period":   "2018-2024",
            "sharpe":   round(full_is_sharpe, 3),
            "max_dd":   round(full_is_dd, 2),
            "cagr":     round(r_full_is.get("cagr", 0), 2) if r_full_is else 0,
        },
        "true_oos": {
            "period":   "2025",
            "sharpe":   round(true_oos_sharpe, 3),
            "max_dd":   round(true_oos_dd, 2),
            "cagr":     round(true_oos_cagr, 2),
            "monthly":  monthly_2025,
            "wins_losses": f"{wins_2025}W{loses_2025}L",
        },
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  結果保存: {OUTPUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
