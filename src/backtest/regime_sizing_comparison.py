"""
backtest/regime_sizing_comparison.py
レジーム別ポジションサイズ制御の比較検証（IS + OOS 2025）

制御方式: TOPIX(1306.T) の MA200 × 20日ボラティリティ で4象限分類
  上昇安定（MA200上 かつ vol低）: size = 1.0
  上昇高ボラ（MA200上 かつ vol高）: size = 0.5
  下落（MA200下）              : size = 0.25

比較シナリオ:
  no_regime  : 現行（サイズ制御なし / 1.0x 固定）
  regime_2   : MA200のみ 2分類（上昇1.0x / 下落0.25x）
  regime_4   : MA200 + ボラ 4分類（上昇安定1.0x / 上昇高ボラ0.5x / 下落0.25x）

市場ショックはStep Aの結論（composite）を適用。

採用条件:
  IS Sharpe 劣化 < 0.1
  OOS MaxDD 改善 >= 1.5pp
  OOS Sharpe >= 0.50

実行:
  cd C:/ai-trading
  python src/backtest/regime_sizing_comparison.py
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
IS_START  = "2018-01-01"
IS_END    = "2024-12-31"
OOS_START = "2025-01-01"
OOS_END   = "2025-12-31"
FULL_END  = OOS_END

# Step A で採用が確認されたモード
SHOCK_MODE = "composite"

# レジームサイズ設定（scale: MA200上/安定, MA200上/高ボラ, MA200下）
SCENARIOS = {
    "no_regime":  {"up_stable": 1.0, "up_hiVol": 1.0, "down": 1.0},
    "regime_2":   {"up_stable": 1.0, "up_hiVol": 1.0, "down": 0.25},
    "regime_4":   {"up_stable": 1.0, "up_hiVol": 0.5, "down": 0.25},
}

# ボラティリティ閾値: TOPIX 20日リターンの標準偏差の中央値で分割
# 実行時に動的計算

OUTPUT_DIR  = "C:/ai-trading/backtests"
OUTPUT_JSON = os.path.join(OUTPUT_DIR, f"regime_sizing_comparison_{time.strftime('%Y-%m-%d')}.json")


# ------------------------------------------------------------------ #
# レジームスケール計算
# ------------------------------------------------------------------ #
def _compute_regime_scale(
    topix_close: pd.Series,
    scenario_params: dict,
    target_index: pd.DatetimeIndex,
    vol_threshold: float,
) -> np.ndarray:
    """
    各日のポジションスケール係数を返す（0〜1.0）。
    target_index に合わせてアライン。
    """
    ma200  = topix_close.rolling(200, min_periods=1).mean()
    vol20  = topix_close.pct_change().rolling(20, min_periods=10).std()
    above  = topix_close >= ma200
    hi_vol = vol20 >= vol_threshold

    # target_index に合わせる
    scale_ser = pd.Series(index=topix_close.index, dtype=float)
    for dt in topix_close.index:
        ab = bool(above.get(dt, True))
        hv = bool(hi_vol.get(dt, False))
        if ab:
            if hv:
                scale_ser[dt] = scenario_params["up_hiVol"]
            else:
                scale_ser[dt] = scenario_params["up_stable"]
        else:
            scale_ser[dt] = scenario_params["down"]

    # アライン
    out = np.ones(len(target_index), dtype=np.float32)
    idx = target_index.get_indexer(scale_ser.index)
    for i, ti in enumerate(target_index):
        loc = scale_ser.index.get_loc(ti) if ti in scale_ser.index else None
        if loc is not None:
            out[i] = float(scale_ser.iloc[loc])
    return out


# ------------------------------------------------------------------ #
# データロード（1回のみ）
# ------------------------------------------------------------------ #
def load_data():
    cfg = load_strategy_config()
    print("[1/3] ユニバース・価格データ読み込み中（RSR42, 2018-2025）...")
    rsr_syms   = _bt._load_rsr_universe()
    trade_syms = rsr_syms

    universe_raw = download_universe({**rsr_syms}, start=IS_START, end=FULL_END, verbose=False)
    topix_close  = _bt._download_topix(IS_START, FULL_END)

    print("[2/3] RSR計算中...")
    rsr42_prices = {s: universe_raw[s]["df"]["Close"] for s in rsr_syms if s in universe_raw}
    rsr_df = calc_universe_rsr(rsr42_prices)

    print("[3/3] Composite Alpha計算中...")
    trade_prices = {s: universe_raw[s]["df"]["Close"] for s in trade_syms if s in universe_raw}
    alpha_df = _bt.calc_composite_alpha_matrix(trade_prices, window=_bt.COMP_ALPHA_WINDOW)
    alpha_df = alpha_df.shift(1)

    regime_df = _bt._calc_regime(topix_close)
    tech_matrices = _bt._precompute_tech_matrices(universe_raw, list(trade_syms.keys()))

    # ボラ閾値: IS期間（2018-2024）の20日ボラ中央値
    vol20_is = topix_close.pct_change().rolling(20, min_periods=10).std()
    vol20_is = vol20_is.loc[IS_START:IS_END].dropna()
    vol_threshold = float(vol20_is.median())

    print(f"  RSR: {rsr_df.shape[1]}銘柄 / TOPIX ボラ中央値: {vol_threshold:.4f}")
    print(f"  shock_mode={SHOCK_MODE} / min_hold={cfg.risk.min_hold_days}日\n")

    return universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms, tech_matrices, topix_close, cfg, vol_threshold


# ------------------------------------------------------------------ #
# 1シナリオ実行
# ------------------------------------------------------------------ #
def run_one(
    universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms,
    tech_matrices, topix_close, cfg, vol_threshold,
    scenario_name: str, period_start: str, period_end: str,
) -> dict:
    params = SCENARIOS[scenario_name]
    result = _bt.run_scenario(
        scenario          = "BASELINE",
        universe_raw      = universe_raw,
        rsr_df            = rsr_df,
        alpha_df          = alpha_df,
        regime_df         = regime_df,
        trade_syms        = trade_syms,
        rsr_syms          = rsr_syms,
        cfg               = cfg,
        start             = period_start,
        end               = period_end,
        capital           = float(cfg.portfolio.capital),
        verbose           = False,
        tech_matrices     = tech_matrices,
        topix_close       = topix_close,
        min_hold          = int(cfg.risk.min_hold_days),
        market_shock_mode = SHOCK_MODE,
        enable_volume_filter     = True,
        enable_volatility_filter = False,
        enable_atr_filter        = False,
        enable_market_filter     = False,
        regime_scale_series  = _make_regime_series(topix_close, params, vol_threshold),
    )
    return result


def _make_regime_series(topix_close: pd.Series, params: dict, vol_threshold: float) -> pd.Series:
    """各日のスケール係数（pd.Series, index=datetime）"""
    ma200  = topix_close.rolling(200, min_periods=1).mean()
    vol20  = topix_close.pct_change().rolling(20, min_periods=10).std()
    scale  = pd.Series(index=topix_close.index, dtype=float)
    above  = topix_close >= ma200
    hi_vol = vol20 >= vol_threshold
    scale[ above &  hi_vol] = params["up_hiVol"]
    scale[ above & ~hi_vol] = params["up_stable"]
    scale[~above]           = params["down"]
    return scale


# ------------------------------------------------------------------ #
# サマリー表示
# ------------------------------------------------------------------ #
def fmt_row(label: str, r: dict) -> str:
    if not r:
        return f"  {label:14s}  データなし"
    s = r.get("sharpe", 0)
    c = r.get("cagr", 0)
    m = r.get("max_dd", 0)
    h = r.get("avg_simultaneous_holdings", 0)
    return (f"  {label:14s}  Sharpe={s:6.3f}  CAGR={c:+6.1f}%  MaxDD={m:+6.1f}%  avgH={h:.2f}")


# ------------------------------------------------------------------ #
# メイン
# ------------------------------------------------------------------ #
def main():
    print("=" * 72)
    print("  レジーム別ポジションサイズ制御 比較（IS 2018-2024 / OOS 2025）")
    print(f"  ショックモード: {SHOCK_MODE}（Step A 採用）")
    print("=" * 72 + "\n")

    (universe_raw, rsr_df, alpha_df, regime_df,
     trade_syms, rsr_syms, tech_matrices,
     topix_close, cfg, vol_threshold) = load_data()

    results = {}

    for sc_name in SCENARIOS:
        marker = " ← 現行" if sc_name == "no_regime" else ""
        p = SCENARIOS[sc_name]
        print(f"── {sc_name}{marker}  (stable={p['up_stable']}, hiVol={p['up_hiVol']}, down={p['down']}) ──")

        r_is = run_one(universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms,
                       tech_matrices, topix_close, cfg, vol_threshold,
                       sc_name, IS_START, IS_END)
        print(fmt_row("IS (2018-24)", r_is))

        r_oos = run_one(universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms,
                        tech_matrices, topix_close, cfg, vol_threshold,
                        sc_name, OOS_START, OOS_END)
        print(fmt_row("OOS (2025)", r_oos))

        # 月次
        eq = r_oos.get("equity_curve")
        monthly = {}
        if eq is not None and not eq.empty:
            for (yr, mo), grp in eq.groupby([eq.index.year, eq.index.month]):
                if len(grp) < 2:
                    continue
                monthly[f"{yr}-{mo:02d}"] = round(float(grp.iloc[-1] / grp.iloc[0] - 1) * 100, 2)
        wins  = sum(1 for v in monthly.values() if v > 0)
        loses = sum(1 for v in monthly.values() if v <= 0)
        print(f"    月次: {wins}勝{loses}敗  {monthly}\n")

        results[sc_name] = {
            "scenario":    sc_name,
            "params":      p,
            "IS":          {k: v for k, v in r_is.items() if k not in ("equity_curve", "_trades")},
            "OOS":         {k: v for k, v in r_oos.items() if k not in ("equity_curve", "_trades")},
            "monthly_oos": monthly,
        }
        del r_is, r_oos
        collect_and_log_process_memory(f"scenario {sc_name}")

    # ---------------------------------------------------------------- #
    # 比較サマリー
    # ---------------------------------------------------------------- #
    base_is_sharpe = results["no_regime"]["IS"].get("sharpe", 0)
    base_oos_mdd   = results["no_regime"]["OOS"].get("max_dd", 0)
    base_oos_sharpe= results["no_regime"]["OOS"].get("sharpe", 0)

    print("=" * 72)
    print("  比較サマリー")
    print(f"  {'シナリオ':14s}  {'IS Sh':>7}  {'IS DD':>8}  {'OOS Sh':>7}  {'OOS DD':>8}  {'OOS CAGR':>9}  判定")
    print("  " + "-" * 70)
    for sc_name, v in results.items():
        ri = v["IS"]
        ro = v["OOS"]
        sh_diff  = ri.get("sharpe", 0) - base_is_sharpe
        mdd_diff = ro.get("max_dd", 0) - base_oos_mdd
        oos_sh   = ro.get("sharpe", 0)
        marker   = "現行" if sc_name == "no_regime" else (
            "✅" if (sh_diff >= -0.1 and mdd_diff >= 1.5 and oos_sh >= 0.50) else "△"
        )
        print(f"  {sc_name:14s}  {ri.get('sharpe',0):>7.3f}  {ri.get('max_dd',0):>+8.1f}%  "
              f"{oos_sh:>7.3f}  {ro.get('max_dd',0):>+8.1f}%  {ro.get('cagr',0):>+8.1f}%  {marker}")
        if sc_name != "no_regime":
            print(f"  {'  vs 現行':14s}  {sh_diff:>+7.3f}  {'':8s}  "
                  f"{'':7s}  {mdd_diff:>+8.1f}pp  {'':9s}")
    print("=" * 72)

    # 採用判定
    print("\n  【採用判定】")
    print("  採用条件: IS Sharpe 劣化 < 0.1 / OOS MaxDD 改善 >= 1.5pp / OOS Sharpe >= 0.50")
    for sc_name, v in results.items():
        if sc_name == "no_regime":
            continue
        ri = v["IS"]
        ro = v["OOS"]
        sh_diff  = ri.get("sharpe", 0) - base_is_sharpe
        mdd_diff = ro.get("max_dd", 0) - base_oos_mdd
        oos_sh   = ro.get("sharpe", 0)
        ok = sh_diff >= -0.1 and mdd_diff >= 1.5 and oos_sh >= 0.50
        verdict = "✅ 採用候補" if ok else "❌ 不採用"
        print(f"  {sc_name:12s}: {verdict}"
              f"  (IS_sh_diff={sh_diff:+.3f}, OOS_MaxDD_diff={mdd_diff:+.1f}pp, OOS_Sharpe={oos_sh:.3f})")

    # ---------------------------------------------------------------- #
    # JSON保存
    # ---------------------------------------------------------------- #
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  結果保存: {OUTPUT_JSON}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
