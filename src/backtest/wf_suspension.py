"""
backtest/wf_suspension.py
Phase 3 Step 2: ルーザー銘柄サスペンション WF検証

DD分解で判明した構造的ルーザー銘柄（7012.T が全損失の53%）を
rolling PnL + 勝率 の組み合わせで一時停止する機能のWF検証。

制御ロジック:
  直近 suspension_lookback_days 営業日の確定PnL が threshold 以下、
  かつ勝率 < win_threshold の場合、suspension_period_days 日間
  新規エントリーを禁止する。

実行:
  cd C:/ai-trading
  python src/backtest/wf_suspension.py
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
SHOCK_MODE = "composite"

# サスペンション設定（グリッドテスト）
SUSPENSION_CONFIGS = [
    {"name": "baseline",       "enable": False, "pnl_thr": 0,         "win_thr": 1.0, "lookback": 90,  "period": 60},
    {"name": "sus_90d_100k",   "enable": True,  "pnl_thr": -100_000,  "win_thr": 0.40, "lookback": 90,  "period": 60},
    {"name": "sus_90d_60k",    "enable": True,  "pnl_thr": -60_000,   "win_thr": 0.40, "lookback": 90,  "period": 60},
    {"name": "sus_60d_60k",    "enable": True,  "pnl_thr": -60_000,   "win_thr": 0.40, "lookback": 60,  "period": 45},
    {"name": "sus_90d_100k_30d","enable": True,  "pnl_thr": -100_000,  "win_thr": 0.40, "lookback": 90,  "period": 30},
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
OUTPUT_JSON = os.path.join(OUTPUT_DIR, f"wf_suspension_{time.strftime('%Y-%m-%d')}.json")

# ------------------------------------------------------------------ #
# データロード
# ------------------------------------------------------------------ #
def load_data(cfg):
    print("[1/2] ユニバース・価格データ読み込み中（RSR42, 2018-2025）...")
    rsr_syms = _bt._load_rsr_universe()
    universe_raw = download_universe({**rsr_syms}, start="2018-01-01", end=DATA_END, verbose=False)
    topix_close  = _bt._download_topix("2018-01-01", DATA_END)

    print("[2/2] RSR・Composite Alpha計算中...")
    rsr42_prices = {s: universe_raw[s]["df"]["Close"] for s in rsr_syms if s in universe_raw}
    rsr_df = calc_universe_rsr(rsr42_prices)
    trade_prices = {s: universe_raw[s]["df"]["Close"] for s in rsr_syms if s in universe_raw}
    alpha_df = _bt.calc_composite_alpha_matrix(trade_prices, window=_bt.COMP_ALPHA_WINDOW)
    alpha_df = alpha_df.shift(1)

    regime_df     = _bt._calc_regime(topix_close)
    tech_matrices = _bt._precompute_tech_matrices(universe_raw, list(rsr_syms.keys()))
    return (universe_raw, rsr_df, alpha_df, regime_df, rsr_syms, tech_matrices, topix_close, cfg)


# ------------------------------------------------------------------ #
# 1セグメント実行
# ------------------------------------------------------------------ #
def run_seg(universe_raw, rsr_df, alpha_df, regime_df, rsr_syms,
            tech_matrices, topix_close, cfg, start: str, end: str,
            sus_cfg: dict) -> dict:
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
        enable_volume_filter     = True,
        enable_volatility_filter = False,
        enable_atr_filter        = False,
        enable_market_filter     = False,
        enable_suspension        = sus_cfg["enable"],
        suspension_pnl_threshold = sus_cfg["pnl_thr"],
        suspension_win_threshold = sus_cfg["win_thr"],
        suspension_lookback_days = sus_cfg["lookback"],
        suspension_period_days   = sus_cfg["period"],
    )


# ------------------------------------------------------------------ #
# サスペンション設定1つのWF全セグ実行
# ------------------------------------------------------------------ #
def run_wf_for_config(sus_cfg, universe_raw, rsr_df, alpha_df, regime_df,
                      rsr_syms, tech_matrices, topix_close, cfg):
    name = sus_cfg["name"]
    seg_results = []

    for seg_def in WF_SEGS:
        n = seg_def["seg"]
        is_s, is_e = seg_def["is"]
        oos_s, oos_e = seg_def["oos"]

        r_is  = run_seg(universe_raw, rsr_df, alpha_df, regime_df, rsr_syms,
                        tech_matrices, topix_close, cfg, is_s, is_e, sus_cfg)
        r_oos = run_seg(universe_raw, rsr_df, alpha_df, regime_df, rsr_syms,
                        tech_matrices, topix_close, cfg, oos_s, oos_e, sus_cfg)

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

    # Full IS
    r_is_full = run_seg(universe_raw, rsr_df, alpha_df, regime_df, rsr_syms,
                        tech_matrices, topix_close, cfg, *IS_FULL, sus_cfg)
    is_full_sh = r_is_full.get("sharpe", 0) if r_is_full else 0.0
    is_full_dd = r_is_full.get("max_dd", 0) if r_is_full else 0.0

    # True OOS 2025
    r_oos25 = run_seg(universe_raw, rsr_df, alpha_df, regime_df, rsr_syms,
                      tech_matrices, topix_close, cfg, *TRUE_OOS, sus_cfg)
    oos25_sh = r_oos25.get("sharpe", 0) if r_oos25 else 0.0
    oos25_dd = r_oos25.get("max_dd", 0) if r_oos25 else 0.0

    oos_sharpes = [s["oos_sharpe"] for s in seg_results]
    pass_count  = sum(1 for s in seg_results if s["win"])
    median_oos  = float(np.median(oos_sharpes))
    worst_dd    = min(s["oos_max_dd"] for s in seg_results)
    avg_ratio   = float(np.mean([s["ratio"] for s in seg_results if s["ratio"] is not None]))

    collect_and_log_process_memory(f"sus_{name}")

    return {
        "config_name":   name,
        "sus_config":    sus_cfg,
        "segments":      seg_results,
        "wf_summary": {
            "pass_count":        f"{pass_count}/5",
            "median_oos_sharpe": round(median_oos, 3),
            "worst_oos_dd":      round(worst_dd, 2),
            "avg_oos_is_ratio":  round(avg_ratio, 3),
        },
        "full_is": {
            "sharpe": round(is_full_sh, 3),
            "max_dd": round(is_full_dd, 2),
        },
        "true_oos_2025": {
            "sharpe": round(oos25_sh, 3),
            "max_dd": round(oos25_dd, 2),
        },
    }


# ------------------------------------------------------------------ #
# メイン
# ------------------------------------------------------------------ #
def main():
    cfg = load_strategy_config()

    print("=" * 72)
    print("  Phase 3 Step 2 WF: ルーザー銘柄サスペンション検証")
    print(f"  shock={SHOCK_MODE}  設定数={len(SUSPENSION_CONFIGS)}")
    print("=" * 72 + "\n")

    (universe_raw, rsr_df, alpha_df, regime_df, rsr_syms,
     tech_matrices, topix_close, cfg) = load_data(cfg)

    all_results = []

    for sus_cfg in SUSPENSION_CONFIGS:
        name = sus_cfg["name"]
        print(f"\n{'─'*60}")
        print(f"  設定: {name}")
        if sus_cfg["enable"]:
            print(f"  PnL閾値={sus_cfg['pnl_thr']:+,.0f}円  勝率<{sus_cfg['win_thr']}"
                  f"  lookback={sus_cfg['lookback']}d  停止={sus_cfg['period']}d")
        else:
            print("  (suspension なし = ベースライン)")

        result = run_wf_for_config(
            sus_cfg, universe_raw, rsr_df, alpha_df, regime_df,
            rsr_syms, tech_matrices, topix_close, cfg
        )
        all_results.append(result)

        # セグ別表示
        for seg in result["segments"]:
            mark = "OK" if seg["win"] else "NG"
            print(f"    Seg{seg['seg']} OOS={seg['oos']}  IS={seg['is_sharpe']:.3f}"
                  f"  OOS={seg['oos_sharpe']:.3f}  DD={seg['oos_max_dd']:+.1f}%  {mark}")

        s = result["wf_summary"]
        fi = result["full_is"]
        t  = result["true_oos_2025"]
        print(f"  WF: {s['pass_count']} wins / median={s['median_oos_sharpe']:.3f}"
              f" / worstDD={s['worst_oos_dd']:+.1f}%")
        print(f"  Full IS: Sharpe={fi['sharpe']:.3f}  MaxDD={fi['max_dd']:+.1f}%")
        print(f"  2025 OOS: Sharpe={t['sharpe']:.3f}  MaxDD={t['max_dd']:+.1f}%")

    # ── 比較サマリー ──────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  比較サマリー")
    print(f"  {'設定':<22}  {'WF勝率':>6}  {'中央値':>7}  {'worstDD':>9}"
          f"  {'IS Sh':>6}  {'IS DD':>7}  {'2025 Sh':>7}  {'2025 DD':>7}")
    print("  " + "-" * 68)
    for r in all_results:
        s = r["wf_summary"]
        fi = r["full_is"]
        t  = r["true_oos_2025"]
        mark = "★" if r["config_name"] != "baseline" and t["sharpe"] > all_results[0]["true_oos_2025"]["sharpe"] else " "
        print(f"  {mark}{r['config_name']:<21}  {s['pass_count']:>6}  "
              f"{s['median_oos_sharpe']:>7.3f}  {s['worst_oos_dd']:>+9.1f}%"
              f"  {fi['sharpe']:>6.3f}  {fi['max_dd']:>+7.1f}%"
              f"  {t['sharpe']:>7.3f}  {t['max_dd']:>+7.1f}%")
    print("=" * 72)

    # JSON 保存
    output = {
        "date":        time.strftime("%Y-%m-%d"),
        "description": "Phase3 Step2: 銘柄サスペンション WF検証",
        "shock_mode":  SHOCK_MODE,
        "results":     all_results,
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  結果保存: {OUTPUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
