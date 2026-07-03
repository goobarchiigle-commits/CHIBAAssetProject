"""
backtest/mom_period_sensitivity.py
RSRモメンタム計算期間（mom_period）感度テスト（IS + OOS 2025 同時検証）

目的:
  2025 OOS でトレンドの持続期間が IS(8日)から OOS(3-5日)に短縮した。
  mom_period=21 は「21日間RSRが上昇」を確認してからエントリーするため、
  3-5日しか続かない相場では天井圏でのエントリーになっている可能性がある。
  → mom_periodを短縮して「エントリータイミングの前出し」効果を検証する。

検証軸:
  mom_period: 5, 7, 10, 14, 21（現行: 21）

評価指標:
  CAGR / Sharpe / MaxDD / avg_hold_days / Profit Factor / R倍率 / 取引回数/年

期間:
  IS  : 2018-01-01 〜 2024-12-31
  OOS : 2025-01-01 〜 2025-12-31

固定パラメータ（確定済み・変更しない）:
  turtle_exit = 55   (2026-03-31確定)
  min_hold    = 3    (2026-03-31確定 / cfg.risk.min_hold_days)
  min_rsr     = 75.0
  max_positions = 3

ユニバース: RSR42統一（2026-03-30）

実行方法:
  cd C:/ai-trading
  python -m src.backtest.mom_period_sensitivity
"""

from __future__ import annotations
import gc
import os, sys, json, warnings, time
from dataclasses import replace

# C:/ai-trading を sys.path に追加
_repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import src.backtest.composite_alpha_bt as _bt
from src.config_loader import load_strategy_config
from src.backtest.rsr import calc_universe_rsr
from src.backtest.universe_builder import download_universe
from src.utils.memory import collect_and_log_process_memory

# OOS 2025 はスナップショット外のため DATA_VERSION を外す
os.environ.pop("DATA_VERSION", None)

# ------------------------------------------------------------------ #
# 設定
# ------------------------------------------------------------------ #
IS_START  = "2018-01-01"
IS_END    = "2024-12-31"
OOS_START = "2025-01-01"
OOS_END   = "2025-12-31"
FULL_END  = OOS_END

MOM_PERIOD_VALUES = [5, 7, 10, 14, 21]
OUTPUT_DIR  = "C:/ai-trading/backtests"
OUTPUT_JSON = os.path.join(OUTPUT_DIR, f"mom_period_sensitivity_{time.strftime('%Y-%m-%d')}.json")


# ------------------------------------------------------------------ #
# データロード（1回だけ実施）
# ------------------------------------------------------------------ #
def load_data():
    print("[1/3] ユニバース・価格データ読み込み中（RSR42, 2018-2025）...")
    rsr_syms   = _bt._load_rsr_universe()
    trade_syms = rsr_syms   # RSR42統一

    all_syms = {**rsr_syms, **trade_syms}
    universe_raw = download_universe(all_syms, start=IS_START, end=FULL_END, verbose=False)
    topix_close  = _bt._download_topix(IS_START, FULL_END)

    print("[2/3] RSR計算中...")
    rsr42_prices = {
        sym: universe_raw[sym]["df"]["Close"]
        for sym in rsr_syms if sym in universe_raw
    }
    rsr_df = calc_universe_rsr(rsr42_prices)

    print("[3/3] Composite Alpha計算中...")
    trade_prices = {
        sym: universe_raw[sym]["df"]["Close"]
        for sym in trade_syms if sym in universe_raw
    }
    alpha_df = _bt.calc_composite_alpha_matrix(trade_prices, window=_bt.COMP_ALPHA_WINDOW)
    alpha_df = alpha_df.shift(1)

    regime_df = _bt._calc_regime(topix_close)

    print(f"  RSR: {rsr_df.shape[1]}銘柄 × {rsr_df.shape[0]}日")
    print(f"  取引対象: {len(trade_syms)}銘柄\n")

    return universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms


# ------------------------------------------------------------------ #
# 1シナリオ実行
# ------------------------------------------------------------------ #
def run_one(
    universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms,
    base_cfg,
    mom_period: int, period_start: str, period_end: str
) -> dict:
    # frozen dataclass なので replace() で mom_period だけ上書き
    cfg = replace(base_cfg, fujiko=replace(base_cfg.fujiko, mom_period=mom_period))

    return _bt.run_scenario(
        scenario      = "BASELINE",
        universe_raw  = universe_raw,
        rsr_df        = rsr_df,
        alpha_df      = alpha_df,
        regime_df     = regime_df,
        trade_syms    = trade_syms,
        rsr_syms      = rsr_syms,
        cfg           = cfg,
        start         = period_start,
        end           = period_end,
        capital       = float(cfg.portfolio.capital),
        verbose       = False,
        min_hold      = cfg.risk.min_hold_days,
    )


# ------------------------------------------------------------------ #
# Profit Factor 計算
# ------------------------------------------------------------------ #
def calc_profit_factor(r: dict) -> float:
    wr  = r.get("win_rate", 0) / 100.0
    wp  = r.get("avg_win_pct",  0)
    lp  = abs(r.get("avg_lose_pct", 0))
    if (1 - wr) < 1e-9 or lp < 1e-9:
        return float("inf")
    return round((wr * wp) / ((1 - wr) * lp), 3)


# ------------------------------------------------------------------ #
# 表示ヘルパー
# ------------------------------------------------------------------ #
def fmt_row(label: str, r: dict) -> str:
    if not r:
        return f"  {label:30s}  データなし"
    cagr  = r.get("cagr",  0)
    sh    = r.get("sharpe", 0)
    mdd   = r.get("max_dd", 0)
    hold  = r.get("avg_hold_days", 0)
    nyr   = r.get("n_trades_yr", 0)
    wr    = r.get("win_rate", 0)
    rm    = r.get("r_multiple", 0)
    pf    = calc_profit_factor(r)
    pf_s  = f"{pf:.2f}" if pf != float("inf") else "∞"
    return (
        f"  {label:30s}  CAGR={cagr:+6.1f}%  Sharpe={sh:5.3f}"
        f"  MaxDD={mdd:6.1f}%  hold={hold:.1f}d"
        f"  tr/yr={nyr:.0f}  WR={wr:.1f}%  R={rm:.2f}  PF={pf_s}"
    )


# ------------------------------------------------------------------ #
# メイン
# ------------------------------------------------------------------ #
def main():
    base_cfg = load_strategy_config()

    print("=" * 80)
    print("  RSRモメンタム期間 感度テスト（IS 2018-2024 / OOS 2025）")
    print("  ユニバース: RSR42統一（42銘柄）/ シナリオ: BASELINE")
    print(f"  検証軸: mom_period = {MOM_PERIOD_VALUES}  （現行: {base_cfg.fujiko.mom_period}）")
    print(f"  固定値: turtle_exit={base_cfg.fujiko.turtle_exit}"
          f"  min_hold={base_cfg.risk.min_hold_days}"
          f"  min_rsr={base_cfg.fujiko.min_rsr}"
          f"  max_positions={base_cfg.portfolio.max_positions}"
          f"  capital={base_cfg.portfolio.capital:,}")
    print("=" * 80 + "\n")

    universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms = load_data()

    results = {}

    for mp in MOM_PERIOD_VALUES:
        key    = f"mom{mp}d"
        marker = " ← 現行" if mp == base_cfg.fujiko.mom_period else ""
        print(f"── mom_period={mp}日{marker} ──────────────────────────")

        r_is = run_one(universe_raw, rsr_df, alpha_df, regime_df,
                       trade_syms, rsr_syms, base_cfg, mp, IS_START, IS_END)
        print(fmt_row("IS  (2018-2024)", r_is))

        r_oos = run_one(universe_raw, rsr_df, alpha_df, regime_df,
                        trade_syms, rsr_syms, base_cfg, mp, OOS_START, OOS_END)
        print(fmt_row("OOS (2025)",     r_oos))
        print()

        results[key] = {
            "mom_period":        mp,
            "profit_factor_IS":  calc_profit_factor(r_is),
            "profit_factor_OOS": calc_profit_factor(r_oos),
            "IS":                r_is,
            "OOS":               r_oos,
        }
        del r_is, r_oos
        collect_and_log_process_memory(f"scenario {key}")

    # ---------------------------------------------------------------- #
    # サマリーテーブル（OOS）
    # ---------------------------------------------------------------- #
    print("\n" + "=" * 80)
    print("  サマリー（OOS 2025）")
    print(f"  {'設定':10s}  {'CAGR':>8}  {'Sharpe':>7}  {'MaxDD':>7}  "
          f"{'hold':>6}  {'tr/yr':>6}  {'WR':>6}  {'R倍率':>6}  {'PF':>7}")
    print("  " + "-" * 78)
    for key, v in results.items():
        r = v["OOS"]
        marker = " ←現行" if v["mom_period"] == base_cfg.fujiko.mom_period else ""
        if not r:
            print(f"  {key:10s}  データなし")
            continue
        pf = v["profit_factor_OOS"]
        pf_s = f"{pf:.3f}" if pf != float("inf") else "∞"
        print(
            f"  {key:10s}  {r.get('cagr',0):>+7.1f}%  {r.get('sharpe',0):>7.3f}"
            f"  {r.get('max_dd',0):>+7.1f}%  {r.get('avg_hold_days',0):>5.1f}d"
            f"  {r.get('n_trades_yr',0):>6.0f}  {r.get('win_rate',0):>5.1f}%"
            f"  {r.get('r_multiple',0):>6.2f}  {pf_s:>7}{marker}"
        )

    # ---------------------------------------------------------------- #
    # サマリーテーブル（IS）
    # ---------------------------------------------------------------- #
    print("\n  サマリー（IS 2018-2024）")
    print(f"  {'設定':10s}  {'CAGR':>8}  {'Sharpe':>7}  {'MaxDD':>7}  "
          f"{'hold':>6}  {'tr/yr':>6}  {'WR':>6}  {'R倍率':>6}  {'PF':>7}")
    print("  " + "-" * 78)
    for key, v in results.items():
        r = v["IS"]
        marker = " ←現行" if v["mom_period"] == base_cfg.fujiko.mom_period else ""
        if not r:
            print(f"  {key:10s}  データなし")
            continue
        pf = v["profit_factor_IS"]
        pf_s = f"{pf:.3f}" if pf != float("inf") else "∞"
        print(
            f"  {key:10s}  {r.get('cagr',0):>+7.1f}%  {r.get('sharpe',0):>7.3f}"
            f"  {r.get('max_dd',0):>+7.1f}%  {r.get('avg_hold_days',0):>5.1f}d"
            f"  {r.get('n_trades_yr',0):>6.0f}  {r.get('win_rate',0):>5.1f}%"
            f"  {r.get('r_multiple',0):>6.2f}  {pf_s:>7}{marker}"
        )

    # ---------------------------------------------------------------- #
    # 読み方ガイド
    # ---------------------------------------------------------------- #
    print("\n  【読み方】")
    print("  ・hold が mom_period に比例して増減 → エントリータイミング仮説を支持")
    print("  ・OOS Sharpe が単調改善             → 短期化が有効（即採用検討）")
    print("  ・OOS Sharpe がU字型                → 最適点が中間（例: mom10d）")
    print("  ・IS が大幅悪化                     → 過去相場に合わない")
    print("  ・PF > 1.5 を採用基準として参考に")

    # ---------------------------------------------------------------- #
    # JSON保存
    # ---------------------------------------------------------------- #
    def strip_nondumpable(d):
        if isinstance(d, dict):
            return {k: strip_nondumpable(v) for k, v in d.items()
                    if k not in ("_trades", "equity_curve")}
        if isinstance(d, float) and (np.isnan(d) or np.isinf(d)):
            return None
        return d

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(strip_nondumpable(results), f, ensure_ascii=False, indent=2)
    print(f"\n  結果保存: {OUTPUT_JSON}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
