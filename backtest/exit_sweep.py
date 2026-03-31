"""
Exit パラメータ sweep バックテスト
RSR_exit × min_hold の15ケース（+ STEP5ベースライン）を検証する

使い方:
  python -m backtest.exit_sweep           # 15ケース全部
  python -m backtest.exit_sweep --test-a  # テストAのみ (min_hold=3, rsr=72)
"""
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding='utf-8')

from backtest.composite_alpha_bt import (
    _load_rsr_universe, _download_topix, _calc_regime,
    _calc_breadth, _precompute_tech_matrices, run_scenario,
    START, END, CAPITAL, COMP_ALPHA_WINDOW, MIN_RSR,
)
from backtest.rsr import calc_universe_rsr, calc_composite_alpha_matrix
from backtest.universe_builder import download_universe


def load_data():
    """データ一括取得（共通）"""
    print("データ取得中...")
    rsr_syms     = _load_rsr_universe()
    trade_syms   = rsr_syms   # rsr42-trade モード
    universe_raw = download_universe({**rsr_syms}, start=START, end=END, verbose=False)
    topix_close  = _download_topix(START, END)

    rsr42_prices = {sym: universe_raw[sym]["df"]["Close"]
                    for sym in rsr_syms if sym in universe_raw}
    rsr_df = calc_universe_rsr(rsr42_prices)

    trade_prices = {sym: universe_raw[sym]["df"]["Close"]
                    for sym in trade_syms if sym in universe_raw}
    alpha_df = calc_composite_alpha_matrix(trade_prices, window=COMP_ALPHA_WINDOW)
    alpha_df = alpha_df.shift(1)

    regime_df     = _calc_regime(topix_close)
    tech_matrices = _precompute_tech_matrices(universe_raw, list(trade_syms.keys()))

    return universe_raw, rsr_df, alpha_df, regime_df, tech_matrices, rsr_syms, trade_syms


def run_case(label, universe_raw, rsr_df, alpha_df, regime_df, tech_matrices,
             rsr_syms, trade_syms, rsr_exit, min_hold):
    """1ケース実行"""
    res = run_scenario(
        scenario            = "STEP5",
        universe_raw        = universe_raw,
        rsr_df              = rsr_df,
        alpha_df            = alpha_df,
        regime_df           = regime_df,
        tech_matrices       = tech_matrices,
        rsr_syms            = rsr_syms,
        trade_syms          = trade_syms,
        verbose             = False,
        rsr_exit_threshold  = rsr_exit,
        min_hold            = min_hold,
    )
    trades = res.get("_trades", [])
    valid  = [t for t in trades if "exit_idx" in t and "entry_idx" in t]
    hold_arr = np.array([t["exit_idx"] - t["entry_idx"] for t in valid]) if valid else np.array([0])

    rc = res.get("exit_reason_counts", {})
    n  = max(sum(rc.values()), 1)
    rsr_pct   = rc.get("RSR_EXIT", 0) / n * 100
    trail_pct = rc.get("TRAIL_EXIT", 0) / n * 100
    time_pct  = rc.get("TIME_STOP", 0) / n * 100
    strat_pct = rc.get("STRATEGY_EXIT", 0) / n * 100

    return {
        "label":        label,
        "rsr_exit":     rsr_exit,
        "min_hold":     min_hold,
        "cagr":         res["cagr"],
        "sharpe":       res["sharpe"],
        "max_dd":       res["max_dd"],
        "calmar":       res["calmar"],
        "n_trades_yr":  res["n_trades_yr"],
        "n_trades":     res["n_trades"],
        "avg_hold":     round(hold_arr.mean(), 1),
        "avg_exp":      res["avg_exposure"],
        "r_multiple":   res["r_multiple"],
        "avg_win_pct":  res["avg_win_pct"],
        "avg_lose_pct": res["avg_lose_pct"],
        "rsr_pct":      round(rsr_pct, 1),
        "trail_pct":    round(trail_pct, 1),
        "time_pct":     round(time_pct, 1),
        "strat_pct":    round(strat_pct, 1),
        "annual":       res["annual_returns"],
    }


def print_results(results, baseline):
    col_w = 10
    cases = [r["label"] for r in results]

    print("\n" + "=" * 90)
    print("  Exit sweep 結果（STEP5ベース / rsr42-trade）")
    print("=" * 90)

    metrics = [
        ("CAGR(%)",     "cagr",        "{:+.1f}"),
        ("Sharpe",      "sharpe",      "{:.3f}"),
        ("MaxDD(%)",    "max_dd",      "{:.1f}"),
        ("Calmar",      "calmar",      "{:.3f}"),
        ("取引/年",     "n_trades_yr", "{:.0f}"),
        ("avg_hold日",  "avg_hold",    "{:.1f}"),
        ("avgExp%",     "avg_exp",     "{:.1f}"),
        ("R倍率",       "r_multiple",  "{:.2f}"),
        ("avg_win%",    "avg_win_pct", "{:+.2f}"),
        ("avg_lose%",   "avg_lose_pct","{:.2f}"),
        ("RSR_exit%",   "rsr_pct",     "{:.1f}"),
        ("TRAIL%",      "trail_pct",   "{:.1f}"),
        ("TIME%",       "time_pct",    "{:.1f}"),
        ("STRAT%",      "strat_pct",   "{:.1f}"),
    ]

    # ヘッダー
    hdr = f"  {'指標':<14}" + "".join(f" {c:>{col_w}}" for c in cases)
    print(hdr)
    print(f"  {'-'*14}" + f" {'-'*col_w}" * len(cases))

    for label, key, fmt in metrics:
        row = f"  {label:<14}"
        for r in results:
            val = r[key]
            row += f" {fmt.format(val):>{col_w}}"
        print(row)

    # 年次リターン
    print(f"\n  {'年':<6}" + "".join(f" {c:>{col_w}}" for c in cases))
    all_years = sorted({yr for r in results for yr in r["annual"]})
    for yr in all_years:
        row = f"  {yr:<6}"
        for r in results:
            v = r["annual"].get(str(yr), 0.0)
            row += f" {v:>+{col_w}.1f}%"[:-1] + "%"
        print(row)

    # 判定
    print(f"\n  判定（vs STEP5ベースライン: CAGR{baseline['cagr']:+.1f}% / Sharpe{baseline['sharpe']:.3f} / "
          f"Calmar{baseline['calmar']:.3f} / avg_hold{baseline['avg_hold']:.1f}日）")
    for r in results:
        if r["label"] == "STEP5_base":
            continue
        d_cagr   = r["cagr"]   - baseline["cagr"]
        d_sharpe = r["sharpe"] - baseline["sharpe"]
        d_calmar = r["calmar"] - baseline["calmar"]
        d_hold   = r["avg_hold"] - baseline["avg_hold"]
        verdict  = "✅" if d_sharpe > 0.02 and r["max_dd"] > -20 else \
                   "△" if d_sharpe > 0 else "❌"
        print(f"  {verdict} {r['label']:<20}  CAGR{d_cagr:+.1f}pp  Sharpe{d_sharpe:+.3f}  "
              f"Calmar{d_calmar:+.3f}  hold+{d_hold:.1f}日")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-a", action="store_true", help="テストAのみ実行")
    parser.add_argument("--mh46",   action="store_true", help="rsr75_mh4 / rsr75_mh6 追加2ケース")
    parser.add_argument("--oos",    action="store_true", help="OOSテスト(2018-21/2022-24分割)")
    args = parser.parse_args()

    data = load_data()
    universe_raw, rsr_df, alpha_df, regime_df, tech_matrices, rsr_syms, trade_syms = data

    # ─── ベースライン（STEP5 / 現状設定）─────────────────────────────
    print("\nベースライン（STEP5 / rsr=75 / min_hold=0）実行中...")
    baseline = run_case("STEP5_base", *data,
                        rsr_exit=75.0, min_hold=0)
    results = [baseline]

    if args.test_a:
        # ─── テストA のみ ─────────────────────────────────────────────
        print("テストA（rsr=72 / min_hold=3）実行中...")
        res_a = run_case("TestA_rsr72_mh3", *data,
                         rsr_exit=72.0, min_hold=3)
        results.append(res_a)

    elif args.mh46:
        # ─── 追加2ケース：rsr75_mh4 / rsr75_mh6 ────────────────────
        for mh in [4, 6]:
            label = f"rsr75_mh{mh}"
            print(f"  {label} 実行中...")
            r = run_case(label, *data, rsr_exit=75.0, min_hold=mh)
            results.append(r)
        # 比較用に mh=5 も追加
        print("  rsr75_mh5（比較用）実行中...")
        r = run_case("rsr75_mh5", *data, rsr_exit=75.0, min_hold=5)
        results.append(r)

    elif args.oos:
        # ─── OOS テスト：IS(2018-21) / OOS(2022-24) ──────────────────
        print("\n=== OOSテスト（IS: 2018-2021 / OOS: 2022-2024）===")
        cases = [
            ("STEP5_IS",    75.0, 0, "2018-01-01", "2021-12-31"),
            ("STEP5_OOS",   75.0, 0, "2022-01-01", "2024-12-31"),
            ("mh5_IS",      75.0, 5, "2018-01-01", "2021-12-31"),
            ("mh5_OOS",     75.0, 5, "2022-01-01", "2024-12-31"),
        ]
        results = []  # ベースラインは別途表示
        for label, rsr, mh, s, e in cases:
            print(f"  {label} 実行中...")
            res = run_scenario(
                scenario           = "STEP5",
                universe_raw       = universe_raw,
                rsr_df             = rsr_df,
                alpha_df           = alpha_df,
                regime_df          = regime_df,
                tech_matrices      = tech_matrices,
                rsr_syms           = rsr_syms,
                trade_syms         = trade_syms,
                verbose            = False,
                start              = s,
                end                = e,
                rsr_exit_threshold = rsr,
                min_hold           = mh,
            )
            trades = res.get("_trades", [])
            valid  = [t for t in trades if "exit_idx" in t and "entry_idx" in t]
            hold_arr = np.array([t["exit_idx"] - t["entry_idx"]
                                 for t in valid]) if valid else np.array([0])
            rc  = res.get("exit_reason_counts", {})
            n   = max(sum(rc.values()), 1)
            results.append({
                "label":        label,
                "rsr_exit":     rsr,
                "min_hold":     mh,
                "cagr":         res["cagr"],
                "sharpe":       res["sharpe"],
                "max_dd":       res["max_dd"],
                "calmar":       res["calmar"],
                "n_trades_yr":  res["n_trades_yr"],
                "n_trades":     res["n_trades"],
                "avg_hold":     round(hold_arr.mean(), 1),
                "avg_exp":      res["avg_exposure"],
                "r_multiple":   res["r_multiple"],
                "avg_win_pct":  res["avg_win_pct"],
                "avg_lose_pct": res["avg_lose_pct"],
                "rsr_pct":      round(rc.get("RSR_EXIT", 0) / n * 100, 1),
                "trail_pct":    round(rc.get("TRAIL_EXIT", 0) / n * 100, 1),
                "time_pct":     round(rc.get("TIME_STOP", 0) / n * 100, 1),
                "strat_pct":    round(rc.get("STRATEGY_EXIT", 0) / n * 100, 1),
                "annual":       res["annual_returns"],
            })
        print_results(results, results[0])  # IS基準で比較
        return

    else:
        # ─── 15ケース グリッド ────────────────────────────────────────
        rsr_vals  = [75, 72, 70, 68, 65]
        hold_vals = [0, 3, 5]
        total = len(rsr_vals) * len(hold_vals)
        done  = 0
        for rsr in rsr_vals:
            for mh in hold_vals:
                done += 1
                label = f"rsr{rsr}_mh{mh}"
                print(f"  [{done:2d}/{total}] {label} 実行中...")
                r = run_case(label, *data, rsr_exit=float(rsr), min_hold=mh)
                results.append(r)

    print_results(results, baseline)


if __name__ == "__main__":
    main()
