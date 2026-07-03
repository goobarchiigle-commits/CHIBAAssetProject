"""
backtest/wf_regime_sym.py
Step 3: 個別銘柄モメンタム連動サイジング WF検証

制御ロジック:
  TOPIX >= MA200: size = 1.0（通常）
  TOPIX < MA200 + symbol_mom_20 > 0: size = 0.75（弱市場・個別強）
  TOPIX < MA200 + symbol_mom_20 <= 0: size = 0.40（弱市場・個別弱）

2022年の失敗（TOPIX指数ベースで一律削減）を修正する本質改善。
market_shock_mode = composite を組み合わせる。

実行:
  cd C:/ai-trading
  python src/backtest/wf_regime_sym.py
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
# 設定
# ------------------------------------------------------------------ #
SHOCK_MODE  = "composite"
SCALE_UP    = 0.75   # TOPIX < MA200 かつ 個別 mom20 > 0
SCALE_DOWN  = 0.40   # TOPIX < MA200 かつ 個別 mom20 <= 0
MOM_PERIOD  = 20

WF_SEGS = [
    {"seg": 1, "is": ("2018-01-01", "2019-12-31"), "oos": ("2020-01-01", "2020-12-31")},
    {"seg": 2, "is": ("2019-01-01", "2020-12-31"), "oos": ("2021-01-01", "2021-12-31")},
    {"seg": 3, "is": ("2020-01-01", "2021-12-31"), "oos": ("2022-01-01", "2022-12-31")},
    {"seg": 4, "is": ("2021-01-01", "2022-12-31"), "oos": ("2023-01-01", "2023-12-31")},
    {"seg": 5, "is": ("2022-01-01", "2023-12-31"), "oos": ("2024-01-01", "2024-12-31")},
]
TRUE_OOS = ("2025-01-01", "2025-12-31")
DATA_END = "2025-12-31"

OUTPUT_DIR  = "C:/ai-trading/backtests"
OUTPUT_JSON = os.path.join(OUTPUT_DIR, f"wf_regime_sym_{time.strftime('%Y-%m-%d')}.json")

# 採用基準
PASS_WIN_MIN    = 5
MEDIAN_OOS_MIN  = 0.55
WORST_DD_LIMIT  = -12.0
OOS25_SH_MIN    = 0.55


# ------------------------------------------------------------------ #
# データロード
# ------------------------------------------------------------------ #
def load_data(cfg):
    print("[1/3] ユニバース・価格データ読み込み中（RSR42, 2018-2025）...")
    rsr_syms = _bt._load_rsr_universe()
    universe_raw = download_universe({**rsr_syms}, start="2018-01-01", end=DATA_END, verbose=False)
    topix_close  = _bt._download_topix("2018-01-01", DATA_END)

    print("[2/3] RSR・Composite Alpha計算中...")
    rsr42_prices = {s: universe_raw[s]["df"]["Close"] for s in rsr_syms if s in universe_raw}
    rsr_df = calc_universe_rsr(rsr42_prices)
    trade_prices = {s: universe_raw[s]["df"]["Close"] for s in rsr_syms if s in universe_raw}
    alpha_df = _bt.calc_composite_alpha_matrix(trade_prices, window=_bt.COMP_ALPHA_WINDOW)
    alpha_df = alpha_df.shift(1)

    regime_df     = _bt._calc_regime(topix_close)
    tech_matrices = _bt._precompute_tech_matrices(universe_raw, list(rsr_syms.keys()))

    # ── sym_regime_df 構築 ──────────────────────────────────────────
    # TOPIX MA200 vs 各銘柄 20日モメンタム の組み合わせでスケール決定
    topix_ma200 = topix_close.rolling(200, min_periods=1).mean()
    above_ma200 = topix_close >= topix_ma200

    all_close = pd.DataFrame({
        s: universe_raw[s]["df"]["Close"]
        for s in rsr_syms if s in universe_raw
    })
    # shift(1): 当日の open 判断に前日 close を使う（先読み防止）
    sym_mom20 = all_close.pct_change(MOM_PERIOD).shift(1)

    # 初期値 1.0（TOPIX上 or データ不足の場合）
    sym_regime = pd.DataFrame(1.0, index=sym_mom20.index, columns=sym_mom20.columns)

    # TOPIX MA200下の日: 個別モメンタムでスケール分岐
    bear_dates = above_ma200[~above_ma200].index
    bear_dates_in_idx = sym_regime.index.intersection(bear_dates)
    for dt in bear_dates_in_idx:
        mom_row = sym_mom20.loc[dt]
        sym_regime.loc[dt, mom_row > 0]            = SCALE_UP
        sym_regime.loc[dt, mom_row <= 0]           = SCALE_DOWN
        sym_regime.loc[dt, mom_row.isna()]         = SCALE_DOWN   # NaN -> 保守的に縮小

    bear_days = len(bear_dates_in_idx)
    print(f"  TOPIX MA200下: {bear_days}日 / "
          f"scale: 上昇={SCALE_UP}x 弱={SCALE_DOWN}x mom_period={MOM_PERIOD}d\n")

    return (universe_raw, rsr_df, alpha_df, regime_df, rsr_syms,
            tech_matrices, topix_close, sym_regime, cfg)


# ------------------------------------------------------------------ #
# 1セグメント実行
# ------------------------------------------------------------------ #
def run_seg(universe_raw, rsr_df, alpha_df, regime_df, rsr_syms,
            tech_matrices, topix_close, sym_regime, cfg,
            start: str, end: str) -> dict:
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
        market_shock_mode = SHOCK_MODE,
        sym_regime_df    = sym_regime,
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

    print("=" * 72)
    print("  Step 3 WF: 個別銘柄モメンタム連動サイジング")
    print(f"  shock={SHOCK_MODE} / TOPIX<MA200 -> mom>0:{SCALE_UP}x / mom<=0:{SCALE_DOWN}x")
    print(f"  採用基準: {PASS_WIN_MIN}/5 / median>={MEDIAN_OOS_MIN} / worst DD>{WORST_DD_LIMIT}% / 2025>={OOS25_SH_MIN}")
    print("=" * 72 + "\n")

    (universe_raw, rsr_df, alpha_df, regime_df, rsr_syms,
     tech_matrices, topix_close, sym_regime, cfg) = load_data(cfg)

    seg_results = []

    for seg_def in WF_SEGS:
        n = seg_def["seg"]
        is_s, is_e = seg_def["is"]
        oos_s, oos_e = seg_def["oos"]

        r_is  = run_seg(universe_raw, rsr_df, alpha_df, regime_df, rsr_syms,
                        tech_matrices, topix_close, sym_regime, cfg, is_s, is_e)
        r_oos = run_seg(universe_raw, rsr_df, alpha_df, regime_df, rsr_syms,
                        tech_matrices, topix_close, sym_regime, cfg, oos_s, oos_e)

        is_sh  = r_is.get("sharpe",  0) if r_is  else 0.0
        oos_sh = r_oos.get("sharpe", 0) if r_oos else 0.0
        oos_dd = r_oos.get("max_dd", 0) if r_oos else 0.0
        ratio  = round(oos_sh / is_sh, 3) if is_sh != 0 else None
        win    = oos_sh > 0

        seg_results.append({
            "seg": n, "is": f"{is_s[:4]}-{is_e[:4]}",
            "oos": f"{oos_s[:4]}-{oos_e[:4]}",
            "is_sharpe": round(is_sh, 3), "oos_sharpe": round(oos_sh, 3),
            "oos_max_dd": round(oos_dd, 2), "ratio": ratio, "win": win,
        })

        mark = "OK" if win else "NG"
        print(f"  Seg{n} OOS={oos_s[:4]}  IS={is_sh:.3f}  OOS={oos_sh:.3f}"
              f"  DD={oos_dd:+.1f}%  ratio={ratio}  {mark}")

        del r_is, r_oos
        collect_and_log_process_memory(f"seg{n}")

    # True OOS
    print("\n  True OOS 2025...")
    r_oos25 = run_seg(universe_raw, rsr_df, alpha_df, regime_df, rsr_syms,
                      tech_matrices, topix_close, sym_regime, cfg, *TRUE_OOS)
    oos25_sh = r_oos25.get("sharpe", 0) if r_oos25 else 0.0
    oos25_dd = r_oos25.get("max_dd", 0) if r_oos25 else 0.0
    oos25_cg = r_oos25.get("cagr",   0) if r_oos25 else 0.0

    monthly = {}
    eq = r_oos25.get("equity_curve") if r_oos25 else None
    if eq is not None and not eq.empty:
        for (yr, mo), grp in eq.groupby([eq.index.year, eq.index.month]):
            if len(grp) < 2:
                continue
            monthly[f"{yr}-{mo:02d}"] = round(float(grp.iloc[-1] / grp.iloc[0] - 1) * 100, 2)

    oos_sharpes = [s["oos_sharpe"] for s in seg_results]
    pass_count  = sum(1 for s in seg_results if s["win"])
    median_oos  = float(np.median(oos_sharpes))
    worst_dd    = min(s["oos_max_dd"] for s in seg_results)
    avg_ratio   = float(np.mean([s["ratio"] for s in seg_results if s["ratio"] is not None]))

    ok_pass   = pass_count >= PASS_WIN_MIN
    ok_median = median_oos >= MEDIAN_OOS_MIN
    ok_dd     = worst_dd   >= WORST_DD_LIMIT
    ok_oos25  = oos25_sh   >= OOS25_SH_MIN
    verdict = "PASS" if (ok_pass and ok_median and ok_dd and ok_oos25) else "FAIL"

    print("\n" + "=" * 72)
    print(f"  OOS 勝率   : {pass_count}/5  {'OK' if ok_pass else 'NG'}")
    print(f"  OOS 中央値 : {median_oos:.3f}  (>= {MEDIAN_OOS_MIN})  {'OK' if ok_median else 'NG'}")
    print(f"  worst DD   : {worst_dd:+.1f}%  (> {WORST_DD_LIMIT}%)  {'OK' if ok_dd else 'NG'}")
    print(f"  2025 Sharpe: {oos25_sh:.3f}  (>= {OOS25_SH_MIN})  {'OK' if ok_oos25 else 'NG'}")
    print(f"  2025 MaxDD : {oos25_dd:+.1f}%  CAGR={oos25_cg:+.1f}%")
    print(f"  avg OOS/IS : {avg_ratio:.3f}")
    print(f"  月次 2025  : {monthly}")
    print(f"\n  総合判定: {verdict}")
    print("=" * 72)

    output = {
        "date":        time.strftime("%Y-%m-%d"),
        "description": f"Step3 sym_regime: up={SCALE_UP} down={SCALE_DOWN} mom={MOM_PERIOD}d",
        "config": {
            "shock_exit_mode": SHOCK_MODE, "regime_mode": "regime_sym",
            "scale_up": SCALE_UP, "scale_down": SCALE_DOWN, "mom_period": MOM_PERIOD,
            "min_hold": cfg.risk.min_hold_days,
        },
        "wf_segments": seg_results,
        "wf_summary": {
            "oos_wins":          f"{pass_count}/5",
            "median_oos_sharpe": round(median_oos, 3),
            "worst_oos_dd":      round(worst_dd, 2),
            "avg_oos_is_ratio":  round(avg_ratio, 3),
            "judgment":          verdict,
        },
        "true_oos": {
            "period": "2025", "sharpe": round(oos25_sh, 3),
            "max_dd": round(oos25_dd, 2), "cagr": round(oos25_cg, 2),
            "monthly": monthly,
        },
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  結果保存: {OUTPUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
