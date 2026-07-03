"""
backtest/walkforward_revalidation.py
新パラメータ（turtle_exit=55 / min_hold=3）のウォークフォワード検証

目的:
  turtle_exit=55 + min_hold=3 の採用後、
  IS期間（2018-2024）を5セグメントに分割して
  OOS/IS Sharpe比を計算し過学習を確認する。

  また真のOOS（2025年）もあわせて確認。

検証構造（ローリングウィンドウ）:
  Seg1: IS 2018-2019（2年）→ OOS 2020（1年）
  Seg2: IS 2019-2020（2年）→ OOS 2021（1年）
  Seg3: IS 2020-2021（2年）→ OOS 2022（1年）
  Seg4: IS 2021-2022（2年）→ OOS 2023（1年）
  Seg5: IS 2022-2023（2年）→ OOS 2024（1年）

  + 真のOOS 2025（full IS 2018-2024 後）

判断基準:
  OOS勝率 ≥ 4/5 → 過学習なし
  avg OOS/IS Sharpe比 ≥ 0.7 → 良好

パラメータ（確定値 2026-03-31）:
  turtle_exit = 55
  min_hold    = 3
  min_rsr     = 75
  max_pos     = 3
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

# OOS 2025はスナップショット外なのでyfinanceを使用
import os as _os
_os.environ.pop("DATA_VERSION", None)

# ------------------------------------------------------------------ #
# 設定（確定値）
# ------------------------------------------------------------------ #
TURTLE_EXIT = 55
MIN_HOLD    = 3

FULL_START  = "2018-01-01"
FULL_END    = "2025-12-31"

# ウォークフォワード5セグメント（ローリング2年IS+1年OOS）
WF_SEGMENTS = [
    ("2018-01-01", "2019-12-31", "2020-01-01", "2020-12-31"),
    ("2019-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
    ("2020-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
    ("2021-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
    ("2022-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
]

# フル IS + 真のOOS
FULL_IS_START  = "2018-01-01"
FULL_IS_END    = "2024-12-31"
TRUE_OOS_START = "2025-01-01"
TRUE_OOS_END   = "2025-12-31"

OUTPUT_DIR  = "C:/ai-trading/backtests"
OUTPUT_JSON = os.path.join(OUTPUT_DIR, f"walkforward_revalidation_{time.strftime('%Y-%m-%d')}.json")
LOG_FILE    = f"C:/ai-trading/logs/research/walkforward_revalidation_{time.strftime('%Y-%m-%d')}.log"


# ------------------------------------------------------------------ #
# データロード
# ------------------------------------------------------------------ #
def load_data():
    print(f"[1/3] ユニバース・価格データ読み込み中（RSR42, {FULL_START}〜{FULL_END}）...")
    rsr_syms   = _bt._load_rsr_universe()
    trade_syms = rsr_syms

    all_syms     = {**rsr_syms, **trade_syms}
    universe_raw = download_universe(all_syms, start=FULL_START, end=FULL_END, verbose=False)
    topix_close  = _bt._download_topix(FULL_START, FULL_END)

    print("[2/3] RSR計算中...")
    rsr42_prices = {sym: universe_raw[sym]["df"]["Close"] for sym in rsr_syms if sym in universe_raw}
    rsr_df = calc_universe_rsr(rsr42_prices)

    print("[3/3] Composite Alpha計算中...")
    trade_prices = {sym: universe_raw[sym]["df"]["Close"] for sym in trade_syms if sym in universe_raw}
    alpha_df = _bt.calc_composite_alpha_matrix(trade_prices, window=_bt.COMP_ALPHA_WINDOW)
    alpha_df = alpha_df.shift(1)

    regime_df = _bt._calc_regime(topix_close)

    print(f"  RSR: {rsr_df.shape[1]}銘柄 × {rsr_df.shape[0]}日")
    print(f"  TURTLE_EXIT={TURTLE_EXIT}日 / MIN_HOLD={MIN_HOLD}日\n")

    return universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms


# ------------------------------------------------------------------ #
# 1セグメント実行
# ------------------------------------------------------------------ #
def run_period(universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms,
               start: str, end: str) -> dict:
    orig_exit = _bt.TURTLE_EXIT
    _bt.TURTLE_EXIT = TURTLE_EXIT
    try:
        result = _bt.run_scenario(
            scenario     = "BASELINE",
            universe_raw = universe_raw,
            rsr_df       = rsr_df,
            alpha_df     = alpha_df,
            regime_df    = regime_df,
            trade_syms   = trade_syms,
            rsr_syms     = rsr_syms,
            start        = start,
            end          = end,
            capital      = _bt.CAPITAL,
            verbose      = False,
            min_hold     = MIN_HOLD,
        )
    finally:
        _bt.TURTLE_EXIT = orig_exit
    return result or {}


# ------------------------------------------------------------------ #
# メイン
# ------------------------------------------------------------------ #
def main():
    print("=" * 72)
    print("  ウォークフォワード再検証（turtle_exit=55 / min_hold=3）")
    print("  ユニバース: RSR42統一（42銘柄）")
    print("=" * 72 + "\n")

    universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms = load_data()

    # ---------------------------------------------------------------- #
    # ウォークフォワード5セグメント
    # ---------------------------------------------------------------- #
    print("─" * 72)
    print("  ウォークフォワード（5セグメント）")
    print("─" * 72)

    wf_results = []
    for i, (is_s, is_e, oos_s, oos_e) in enumerate(WF_SEGMENTS, 1):
        print(f"\n  Seg{i}: IS {is_s[:4]}-{is_e[:4]}  →  OOS {oos_s[:4]}")

        r_is  = run_period(universe_raw, rsr_df, alpha_df, regime_df,
                           trade_syms, rsr_syms, is_s, is_e)
        r_oos = run_period(universe_raw, rsr_df, alpha_df, regime_df,
                           trade_syms, rsr_syms, oos_s, oos_e)

        is_sharpe  = r_is.get("sharpe", 0)
        oos_sharpe = r_oos.get("sharpe", 0)
        ratio      = oos_sharpe / is_sharpe if is_sharpe != 0 else float("nan")
        win        = oos_sharpe > 0

        print(f"    IS  → CAGR={r_is.get('cagr',0):+.1f}%  Sharpe={is_sharpe:.3f}  "
              f"MaxDD={r_is.get('max_dd',0):.1f}%  hold={r_is.get('avg_hold_days',0):.0f}d")
        print(f"    OOS → CAGR={r_oos.get('cagr',0):+.1f}%  Sharpe={oos_sharpe:.3f}  "
              f"MaxDD={r_oos.get('max_dd',0):.1f}%  hold={r_oos.get('avg_hold_days',0):.0f}d  "
              f"比={ratio:.2f}  {'✅' if win else '❌'}")

        wf_results.append({
            "seg": i,
            "is_period":  f"{is_s}/{is_e}",
            "oos_period": f"{oos_s}/{oos_e}",
            "is_sharpe":  round(is_sharpe, 3),
            "oos_sharpe": round(oos_sharpe, 3),
            "ratio":      round(ratio, 3) if not np.isnan(ratio) else None,
            "win":        bool(win),
            "IS":         r_is,
            "OOS":        r_oos,
        })
        del r_is, r_oos
        gc.collect()

    # ---------------------------------------------------------------- #
    # フルIS + 真のOOS 2025
    # ---------------------------------------------------------------- #
    print(f"\n{'─' * 72}")
    print(f"  フルIS（{FULL_IS_START[:4]}-{FULL_IS_END[:4]}）+ 真のOOS 2025")
    print("─" * 72)

    r_full_is  = run_period(universe_raw, rsr_df, alpha_df, regime_df,
                             trade_syms, rsr_syms, FULL_IS_START, FULL_IS_END)
    r_true_oos = run_period(universe_raw, rsr_df, alpha_df, regime_df,
                             trade_syms, rsr_syms, TRUE_OOS_START, TRUE_OOS_END)

    full_is_sharpe  = r_full_is.get("sharpe", 0)
    true_oos_sharpe = r_true_oos.get("sharpe", 0)
    true_oos_ratio  = true_oos_sharpe / full_is_sharpe if full_is_sharpe != 0 else float("nan")

    print(f"  Full IS  → CAGR={r_full_is.get('cagr',0):+.1f}%  Sharpe={full_is_sharpe:.3f}  "
          f"MaxDD={r_full_is.get('max_dd',0):.1f}%  hold={r_full_is.get('avg_hold_days',0):.0f}d")
    print(f"  OOS 2025 → CAGR={r_true_oos.get('cagr',0):+.1f}%  Sharpe={true_oos_sharpe:.3f}  "
          f"MaxDD={r_true_oos.get('max_dd',0):.1f}%  hold={r_true_oos.get('avg_hold_days',0):.0f}d")

    # ---------------------------------------------------------------- #
    # サマリー
    # ---------------------------------------------------------------- #
    print("\n" + "=" * 72)
    print("  ウォークフォワード サマリー")
    print("=" * 72)

    wins       = sum(1 for r in wf_results if r["win"])
    ratios     = [r["ratio"] for r in wf_results if r["ratio"] is not None]
    avg_ratio  = float(np.mean(ratios)) if ratios else 0.0
    worst_oos  = min(r["oos_sharpe"] for r in wf_results)

    print(f"\n  OOS勝率    : {wins}/5  {'✅ 良好' if wins >= 4 else '⚠ 要確認'}")
    print(f"  OOS/IS比   : {avg_ratio:.3f}  {'✅ 良好' if avg_ratio >= 0.7 else '⚠ 低め'}")
    print(f"  最悪OOS    : Sharpe={worst_oos:.3f}  {'✅' if worst_oos > 0 else '❌'}")
    print(f"\n  Full IS    : CAGR={r_full_is.get('cagr',0):+.1f}%  Sharpe={full_is_sharpe:.3f}  MaxDD={r_full_is.get('max_dd',0):.1f}%")
    print(f"  真OOS 2025 : CAGR={r_true_oos.get('cagr',0):+.1f}%  Sharpe={true_oos_sharpe:.3f}  MaxDD={r_true_oos.get('max_dd',0):.1f}%  比={true_oos_ratio:.3f}")

    print(f"\n  {'Seg':5s}  {'IS期間':18s}  {'OOS期間':12s}  {'IS Sharpe':>10}  {'OOS Sharpe':>11}  {'比':>5}  {'勝敗':>4}")
    print("  " + "-" * 68)
    for r in wf_results:
        ratio_str = f"{r['ratio']:5.2f}" if r['ratio'] is not None else "  N/A"
        print(f"  Seg{r['seg']}  {r['is_period']:18s}  {r['oos_period'][:4]+'-'+r['oos_period'][5:9]:12s}"
              f"  {r['is_sharpe']:>10.3f}  {r['oos_sharpe']:>11.3f}  {ratio_str:>5}  {'✅' if r['win'] else '❌'}")
    print(f"  {'Full':5s}  {'2018-01/2024-12':18s}  {'2025':12s}"
          f"  {full_is_sharpe:>10.3f}  {true_oos_sharpe:>11.3f}  {true_oos_ratio:>5.2f}  {'✅' if true_oos_sharpe > 0 else '❌'}")

    # Phase 2判定
    ph2_ok = wins >= 4 and avg_ratio >= 0.7
    print(f"\n  Phase 2 実運用継続基準: {'✅ クリア' if ph2_ok else '❌ 未達'}")
    print(f"  （OOS勝率≥4/5 かつ OOS/IS比≥0.7）")

    # ---------------------------------------------------------------- #
    # JSON保存
    # ---------------------------------------------------------------- #
    def strip_trades(d):
        if isinstance(d, dict):
            return {k: strip_trades(v) for k, v in d.items() if k not in ("_trades", "equity_curve")}
        return d

    output = {
        "run_date":       time.strftime("%Y-%m-%d"),
        "params":         {"turtle_exit": TURTLE_EXIT, "min_hold": MIN_HOLD},
        "wf_segments":    strip_trades(wf_results),
        "full_is":        strip_trades(r_full_is),
        "true_oos_2025":  strip_trades(r_true_oos),
        "summary": {
            "wins":          wins,
            "avg_ratio":     round(avg_ratio, 3),
            "worst_oos_sharpe": round(worst_oos, 3),
            "phase2_ok":     bool(ph2_ok),
        },
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  結果保存: {OUTPUT_JSON}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
