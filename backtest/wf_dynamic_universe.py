"""
backtest/wf_dynamic_universe.py
Phase 3 Step 1: 動的ユニバース WF検証

DD分解: 7銘柄が全損失の92.5%を占め、構造的ルーザーが固定ユニバースに
残り続けることが根本問題と特定。

実装:
  1. TOPIX100（74銘柄）を母集団とする
  2. 毎月末に3ファクターでスコアリングし Top N を選択
     score = 0.45×mom_63 + 0.30×RS_vs_TOPIX + 0.25×log_volume
  3. 選択銘柄のみ新規エントリーを許可（保有中は継続）
  4. 固定RSR42（ベースライン）と WF比較

先読み防止:
  月 T の選択 = 月 T-1 末時点のデータで計算（当月のデータ不使用）

実行:
  cd C:/ai-trading
  python src/backtest/wf_dynamic_universe.py
"""

from __future__ import annotations
import os, sys, json, warnings, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

import backtest.composite_alpha_bt as _bt
from backtest.rsr import calc_universe_rsr
from backtest.universe_builder import download_universe, TOPIX100_TICKERS
from src.config_loader import load_strategy_config
from src.utils.memory import collect_and_log_process_memory
from src.strategy.universe import (
    build_sym_active_df,
    _zscore,
    MA200_PERIOD,
    MOM_PERIOD,
    VOL_PERIOD,
    BULL_ACTIVE_N,
    BEAR_ACTIVE_N,
    SUSTAINED_BEAR_DAYS,
    LOOKBACK_BEAR_CHECK,
)

os.environ.pop("DATA_VERSION", None)

# ------------------------------------------------------------------ #
# 設定
# ------------------------------------------------------------------ #
SHOCK_MODE  = "composite"
POOL_SYMS   = TOPIX100_TICKERS   # 74銘柄（ベースラインのRSR42を内包）

SCORE_WEIGHTS = {
    # Bull regime
    "bull_mom":  0.40,
    "bull_rsr":  0.35,
    "bull_vol":  0.25,
    # Bear regime
    "bear_rs":   0.50,
    "bear_rsr":  0.30,
    "bear_vol":  0.20,
}

# Bull: Top30 / Bear: Top20（守りの局面で広げすぎない）
ACTIVE_SIZES = [BULL_ACTIVE_N]   # レジームアウェア版は Bull_n=30 で検証

# WF判定基準（2022耐性重視）
PASS_SEG3_2022_MIN = 0.0    # Seg3 2022 OOS > 0 が最優先
PASS_WF_WINS_MIN   = 5      # WF 5/5
PASS_2025_SH_MIN   = 0.80   # 2025 OOS > 0.80

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
OUTPUT_JSON = os.path.join(OUTPUT_DIR, f"wf_dynamic_universe_{time.strftime('%Y-%m-%d')}.json")


# ------------------------------------------------------------------ #
# スコアリング & sym_active_df 構築
# build_sym_active_df / _zscore / 定数は src.strategy.universe から import
# ------------------------------------------------------------------ #
# build_sym_active_df は src.strategy.universe から import 済み


# ------------------------------------------------------------------ #
# データロード
# ------------------------------------------------------------------ #
def load_data(cfg):
    print("[1/3] 全銘柄データロード中（TOPIX100 74銘柄 + RSR42, 2018-2025）...")
    all_syms = {**POOL_SYMS}
    # RSR42 はすでに TOPIX100 に含まれているが念のためマージ
    rsr_syms = _bt._load_rsr_universe()
    for s, sec in rsr_syms.items():
        if s not in all_syms:
            all_syms[s] = sec

    universe_raw = download_universe(all_syms, start="2018-01-01", end=DATA_END, verbose=False)
    topix_close  = _bt._download_topix("2018-01-01", DATA_END)

    print(f"  ロード完了: {len(universe_raw)}銘柄")

    print("[2/3] RSR・Composite Alpha計算中...")
    rsr_syms_avail = {s: sec for s, sec in rsr_syms.items() if s in universe_raw}

    # RSR42のみ（ベースライン比較用・元の挙動を保持）
    rsr42_prices = {s: universe_raw[s]["df"]["Close"] for s in rsr_syms_avail}
    rsr_df_rsr42 = calc_universe_rsr(rsr42_prices)

    # ★ NaN修正: 全プール91銘柄でRSRを計算（動的ユニバース用）
    # → 非RSR42銘柄も RSR 値を持つ = RSR_EXIT が正常に動作する
    all_prices   = {s: universe_raw[s]["df"]["Close"] for s in all_syms if s in universe_raw}
    rsr_df_full  = calc_universe_rsr(all_prices)

    # Composite alpha は RSR42 ベースで維持（ランキング変更なし）
    trade_prices = {s: universe_raw[s]["df"]["Close"] for s in rsr_syms_avail}
    alpha_df = _bt.calc_composite_alpha_matrix(trade_prices, window=_bt.COMP_ALPHA_WINDOW)
    alpha_df = alpha_df.shift(1)

    regime_df     = _bt._calc_regime(topix_close)
    tech_matrices = _bt._precompute_tech_matrices(universe_raw, list(all_syms.keys()))

    print(f"  RSR: RSR42={len(rsr_df_rsr42.columns)}銘柄 / 全プール={len(rsr_df_full.columns)}銘柄（NaN修正済み）")

    # rsr_df_rsr42: ベースライン用（trade_syms=rsr_syms の場合）
    # rsr_df_full:  動的ユニバース用（trade_syms=all_syms の場合）
    return (universe_raw, rsr_df_rsr42, rsr_df_full, alpha_df, regime_df,
            rsr_syms_avail, all_syms, tech_matrices, topix_close, cfg)


# ------------------------------------------------------------------ #
# 1セグメント実行
# ------------------------------------------------------------------ #
def run_seg(universe_raw, rsr_df, alpha_df, regime_df, all_syms,
            tech_matrices, topix_close, cfg,
            start: str, end: str,
            sym_active_df: "pd.DataFrame | None" = None) -> dict:
    return _bt.run_scenario(
        scenario         = "BASELINE",
        universe_raw     = universe_raw,
        rsr_df           = rsr_df,
        alpha_df         = alpha_df,
        regime_df        = regime_df,
        trade_syms       = all_syms,
        rsr_syms         = all_syms,
        cfg              = cfg,
        start            = start,
        end              = end,
        capital          = float(cfg.portfolio.capital),
        verbose          = False,
        tech_matrices    = tech_matrices,
        topix_close      = topix_close,
        min_hold         = int(cfg.risk.min_hold_days),
        market_shock_mode = SHOCK_MODE,
        enable_volume_filter     = True,
        enable_volatility_filter = False,
        enable_atr_filter        = False,
        enable_market_filter     = False,
        sym_active_df    = sym_active_df,
    )


# ------------------------------------------------------------------ #
# 1設定でのWF全セグ実行
# ------------------------------------------------------------------ #
def _build_active(universe_raw, topix_close, all_syms, rsr_df,
                   active_n, bear_n, data_start, end,
                   bear_pool=None):
    """build_sym_active_df の薄いラッパー"""
    if active_n is None:
        return None
    return build_sym_active_df(
        universe_raw, topix_close, list(all_syms.keys()),
        active_n, data_start, end,
        rsr_df_full=rsr_df, bear_n=bear_n, bear_pool=bear_pool,
    )


def run_wf_for_config(config_name, active_n,
                      universe_raw, rsr_df, alpha_df, regime_df, all_syms,
                      tech_matrices, topix_close, cfg,
                      bear_n: int | None = None,
                      bear_pool: list | None = None):
    seg_results = []
    _bear_n    = bear_n if bear_n is not None else (BEAR_ACTIVE_N if active_n else None)
    _bear_pool = bear_pool  # None / list / "bull_score_only"

    print(f"\n{'─'*60}")
    bp_label = "RSR42のみ" if _bear_pool is not None else "全プール"
    label = f"bull={active_n}/bear={_bear_n}({bp_label})" if active_n else "RSR42固定"
    print(f"  設定: {config_name}  ({label})")

    for seg_def in WF_SEGS:
        n = seg_def["seg"]
        is_s, is_e = seg_def["is"]
        oos_s, oos_e = seg_def["oos"]

        active_is  = _build_active(universe_raw, topix_close, all_syms, rsr_df,
                                    active_n, _bear_n, "2017-01-01", is_e, _bear_pool)
        active_oos = _build_active(universe_raw, topix_close, all_syms, rsr_df,
                                    active_n, _bear_n, "2017-01-01", oos_e, _bear_pool)

        r_is  = run_seg(universe_raw, rsr_df, alpha_df, regime_df, all_syms,
                        tech_matrices, topix_close, cfg, is_s, is_e, active_is)
        r_oos = run_seg(universe_raw, rsr_df, alpha_df, regime_df, all_syms,
                        tech_matrices, topix_close, cfg, oos_s, oos_e, active_oos)

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
        mark = "OK" if win else "NG"
        print(f"    Seg{n} OOS={oos_s[:4]}  IS={is_sh:.3f}  OOS={oos_sh:.3f}"
              f"  DD={oos_dd:+.1f}%  {mark}")
        del r_is, r_oos

    # Full IS
    active_full = _build_active(universe_raw, topix_close, all_syms, rsr_df,
                                 active_n, _bear_n, "2017-01-01", IS_FULL[1], _bear_pool)
    r_is_full = run_seg(universe_raw, rsr_df, alpha_df, regime_df, all_syms,
                        tech_matrices, topix_close, cfg, *IS_FULL, active_full)
    is_full_sh = r_is_full.get("sharpe", 0) if r_is_full else 0.0
    is_full_dd = r_is_full.get("max_dd", 0) if r_is_full else 0.0

    # True OOS 2025
    active_oos25 = _build_active(universe_raw, topix_close, all_syms, rsr_df,
                                  active_n, _bear_n, "2017-01-01", TRUE_OOS[1], _bear_pool)
    r_oos25 = run_seg(universe_raw, rsr_df, alpha_df, regime_df, all_syms,
                      tech_matrices, topix_close, cfg, *TRUE_OOS, active_oos25)
    oos25_sh = r_oos25.get("sharpe", 0) if r_oos25 else 0.0
    oos25_dd = r_oos25.get("max_dd", 0) if r_oos25 else 0.0

    oos_sharpes = [s["oos_sharpe"] for s in seg_results]
    pass_count  = sum(1 for s in seg_results if s["win"])
    median_oos  = float(np.median(oos_sharpes))
    worst_dd    = min(s["oos_max_dd"] for s in seg_results)
    valid_ratios = [s["ratio"] for s in seg_results if s["ratio"] is not None]
    avg_ratio   = float(np.mean(valid_ratios)) if valid_ratios else 0.0

    collect_and_log_process_memory(f"dyn_{config_name}")

    s_sum = {
        "pass_count":        f"{pass_count}/5",
        "median_oos_sharpe": round(median_oos, 3),
        "worst_oos_dd":      round(worst_dd, 2),
        "avg_oos_is_ratio":  round(avg_ratio, 3),
    }
    fi = {"sharpe": round(is_full_sh, 3), "max_dd": round(is_full_dd, 2)}
    t  = {"sharpe": round(oos25_sh, 3), "max_dd": round(oos25_dd, 2)}

    print(f"  WF: {s_sum['pass_count']} wins / median={s_sum['median_oos_sharpe']:.3f}"
          f" / worstDD={s_sum['worst_oos_dd']:+.1f}%")
    print(f"  Full IS: Sharpe={fi['sharpe']:.3f}  MaxDD={fi['max_dd']:+.1f}%")
    print(f"  2025 OOS: Sharpe={t['sharpe']:.3f}  MaxDD={t['max_dd']:+.1f}%")

    return {
        "config_name":   config_name,
        "active_n":      active_n,
        "segments":      seg_results,
        "wf_summary":    s_sum,
        "full_is":       fi,
        "true_oos_2025": t,
    }


# ------------------------------------------------------------------ #
# メイン
# ------------------------------------------------------------------ #
def main():
    cfg = load_strategy_config()

    print("=" * 72)
    print("  Phase 3 Step 1 WF: レジームアウェア動的ユニバース")
    print(f"  shock={SHOCK_MODE}  pool=TOPIX100(74銘柄)")
    print(f"  Bull Top{BULL_ACTIVE_N}: mom(0.40)+rsr(0.35)+vol(0.25)")
    print(f"  Bear Top{BEAR_ACTIVE_N}: rs_topix(0.50)+rsr(0.30)+vol(0.20) + rs>0フィルター")
    print(f"  判定: Seg3_2022>{PASS_SEG3_2022_MIN} / WF{PASS_WF_WINS_MIN}/5 / 2025>{PASS_2025_SH_MIN}")
    print("=" * 72 + "\n")

    (universe_raw, rsr_df_rsr42, rsr_df_full, alpha_df, regime_df, rsr_syms, all_syms,
     tech_matrices, topix_close, cfg) = load_data(cfg)

    print("[3/3] WF検証実行中...")

    all_results = []

    # ベースライン: 固定 RSR42 + RSR42専用 RSR（元の設計を保持）
    result_base = run_wf_for_config(
        "baseline_rsr42_fixed", None,
        universe_raw, rsr_df_rsr42, alpha_df, regime_df, rsr_syms,
        tech_matrices, topix_close, cfg
    )
    all_results.append(result_base)

    rsr42_sym_list = list(rsr_syms.keys())

    # 設定A: Bear=全プール + rs>0 フィルター（2022○/2020×）
    all_results.append(run_wf_for_config(
        "dyn_bear_full_rs0", BULL_ACTIVE_N,
        universe_raw, rsr_df_full, alpha_df, regime_df, all_syms,
        tech_matrices, topix_close, cfg,
        bear_n=BEAR_ACTIVE_N, bear_pool=None,
    ))

    # 設定B: Bear=RSR42 + rs>0 フィルター（2020○/2022×）
    all_results.append(run_wf_for_config(
        "dyn_bear_rsr42_rs0", BULL_ACTIVE_N,
        universe_raw, rsr_df_full, alpha_df, regime_df, all_syms,
        tech_matrices, topix_close, cfg,
        bear_n=BEAR_ACTIVE_N, bear_pool=rsr42_sym_list,
    ))

    # 設定C: Bull スコアのまま Bear 時は Top20 に絞るだけ（rs>0 フィルターなし）
    all_results.append(run_wf_for_config(
        "dyn_bear_bull_score_n20", BULL_ACTIVE_N,
        universe_raw, rsr_df_full, alpha_df, regime_df, all_syms,
        tech_matrices, topix_close, cfg,
        bear_n=BEAR_ACTIVE_N, bear_pool="bull_score_only",  # 特殊フラグ
    ))

    # ★ 設定D: RSR42内動的選択 + Bear rs>0フィルター
    # TOPIX100拡張は止め、RSR42（42銘柄）のみで動的選択。
    # alpha_df が RSR42 ベースのため、全選択銘柄で alpha シグナルが有効。
    # Bull: Top30（≈全RSR42）/ Bear(持続型): rs>0 RSR42銘柄のみ Top20
    all_results.append(run_wf_for_config(
        "dyn_rsr42_bear_rs0", BULL_ACTIVE_N,
        universe_raw, rsr_df_rsr42, alpha_df, regime_df, rsr_syms,  # RSR42のみ
        tech_matrices, topix_close, cfg,
        bear_n=BEAR_ACTIVE_N, bear_pool=None,  # pool=None → all_syms(=rsr42) を使用
    ))

    # ★ 設定E: RSR42内動的選択 + Bear rs>0フィルターなし（純スコア Top20）
    all_results.append(run_wf_for_config(
        "dyn_rsr42_bear_score20", BULL_ACTIVE_N,
        universe_raw, rsr_df_rsr42, alpha_df, regime_df, rsr_syms,  # RSR42のみ
        tech_matrices, topix_close, cfg,
        bear_n=BEAR_ACTIVE_N, bear_pool="bull_score_only",
    ))

    # ── 比較サマリー（2022耐性重視判定）──────────────────────────
    print("\n" + "=" * 72)
    print("  比較サマリー（判定: Seg3_2022>0 / WF5/5 / 2025>0.80）")
    print(f"  {'設定':<26}  {'WF勝率':>6}  {'Seg3_22':>8}  {'中央値':>7}"
          f"  {'worstDD':>9}  {'IS Sh':>6}  {'2025 Sh':>7}  {'判定':>6}")
    print("  " + "-" * 76)
    for r in all_results:
        s  = r["wf_summary"]
        fi = r["full_is"]
        t  = r["true_oos_2025"]
        seg3_oos = next((x["oos_sharpe"] for x in r["segments"] if x["seg"] == 3), None)
        ok_seg3  = seg3_oos is not None and seg3_oos > PASS_SEG3_2022_MIN
        ok_wf    = s["pass_count"] == f"{PASS_WF_WINS_MIN}/5"
        ok_2025  = t["sharpe"] >= PASS_2025_SH_MIN
        verdict  = "PASS" if (ok_seg3 and ok_wf and ok_2025) else "FAIL"
        mark = "★" if verdict == "PASS" and r["config_name"] != "baseline_rsr42_fixed" else " "
        seg3_str = f"{seg3_oos:+.3f}" if seg3_oos is not None else "  N/A "
        print(f"  {mark}{r['config_name']:<25}  {s['pass_count']:>6}  {seg3_str:>8}"
              f"  {s['median_oos_sharpe']:>7.3f}  {s['worst_oos_dd']:>+9.1f}%"
              f"  {fi['sharpe']:>6.3f}  {t['sharpe']:>7.3f}  {verdict:>6}")
    print("=" * 72)

    # JSON 保存
    output = {
        "date":          time.strftime("%Y-%m-%d"),
        "description":   "Phase3 Step1: 動的ユニバース WF検証",
        "score_weights": SCORE_WEIGHTS,
        "shock_mode":    SHOCK_MODE,
        "results":       all_results,
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  結果保存: {OUTPUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
