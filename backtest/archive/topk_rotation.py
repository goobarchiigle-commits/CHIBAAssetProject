"""
backtest/topk_rotation.py
Top-k モメンタム・ローテーション戦略 バックテスト

設計（ユーザー指定 2026-03-20）:
  RSR universe  = TEMPORAL24（24銘柄固定）
  ranking       = IBD式12ヶ月複合リターン（calc_composite_return）
  entry         = rank <= k AND slot available
  exit          = rank > k
  max_positions = k
  weight        = equal (capital / k)

テストケース:
  k = 2, 3, 4, 5
  k = 3 + RSモメンタムフィルター（composite_return > TOPIX / RS > 0）

評価指標:
  Sharpe / CAGR / MaxDD / avg_exposure / efficiency (CAGR / avg_exp)
"""

from __future__ import annotations

import os
import sys
import json
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.rcParams["font.family"] = "MS Gothic"

import numpy as np
import pandas as pd
import yfinance as yf

from backtest.universe_builder import download_universe
from backtest.rsr import calc_composite_return
from backtest.engine import TradeCost

UNIVERSE_JSON = "configs/universe/2026Q1_temporal24.json"
IS_START      = "2018-01-01"
IS_END        = "2024-12-31"
OOS_START     = "2025-01-01"
OOS_END       = "2025-12-31"
FULL_END      = "2025-12-31"
CAPITAL       = 2_000_000
OUTPUT        = "C:/Users/owner/.claude/レポート"
BENCH_TICKER  = "1306.T"   # TOPIX ETF


# ================================================================
# バックテストコア
# ================================================================
STOP_LOSS_DEFAULT    = float(os.environ.get("STOP_LOSS",    "-0.15"))
MAX_HOLD_DAYS_DEFAULT = int(os.environ.get("MAX_HOLD_DAYS", "60"))


def _exit_reason(
    pos:           dict,
    current_price: float,
    rank:          float,
    k:             int,
    date:          pd.Timestamp,
    stop_loss:     float | None,
    max_hold_days: int   | None,
) -> str | None:
    """
    ポジション終了理由を返す。None = 保持。

    優先順位:
      1. stop_loss   — 含み損が閾値を超えた
      2. time_exit   — 保有日数が上限を超えた
      3. rank_exit   — ランクが top-k 圏外に落ちた
    """
    unrealized = current_price / pos["entry_px"] - 1.0

    if stop_loss is not None and unrealized <= stop_loss:
        return "stop_loss"

    if max_hold_days is not None:
        hold = (date - pos["entry_date"]).days
        if hold >= max_hold_days:
            return "time_exit"

    if pd.isna(rank) or rank > k:
        return "rank_exit"

    return None


def run_topk(
    prices_dict:      dict[str, pd.Series],
    bench_comp:       pd.Series,
    k:                int,
    momentum_filter:  bool         = False,
    regime_filter:    bool         = False,
    bench_prices:     pd.Series | None = None,
    stop_loss:        float | None = None,
    max_hold_days:    int   | None = None,
    capital:          float        = CAPITAL,
) -> dict:
    """
    Top-k ローテーション バックテスト。

    entry: composite return rank <= k
           (+ composite_return > bench_comp if momentum_filter)
           (+ bench_prices > MA200          if regime_filter)
    exit:  stop_loss OR time_exit OR rank > k  （優先順位順）
    weight: equal (capital / k)

    先読みリーク防止:
      day t 終値でランク判定 → day t+1 始値（Open）で約定
    """
    cost = TradeCost()

    # regime filter 用: TOPIX MA200
    if regime_filter and bench_prices is not None:
        ma200 = bench_prices.rolling(200, min_periods=200).mean()
    else:
        ma200 = None

    # 全銘柄の複合リターン（日次）
    comp_df = pd.DataFrame({
        sym: calc_composite_return(s) for sym, s in prices_dict.items()
    })

    # 日次ランク（1 = 最高リターン）
    rank_df = comp_df.rank(axis=1, ascending=False)

    all_dates = comp_df.index.sort_values()

    positions: dict[str, dict] = {}   # {sym: {qty, cost_basis}}
    cash    = float(capital)
    eq_vals: list[float]       = []
    n_pos:   list[int]         = []
    n_trades = 0

    for i, date in enumerate(all_dates):
        # ---- 時価評価 ----
        mkt = 0.0
        for sym, pos in positions.items():
            if date in prices_dict[sym].index:
                mkt += pos["qty"] * float(prices_dict[sym].loc[date])
        eq_vals.append(cash + mkt)
        n_pos.append(len(positions))

        if i + 1 >= len(all_dates):
            continue
        next_date = all_dates[i + 1]

        if date not in rank_df.index:
            continue

        today_rank = rank_df.loc[date]
        today_comp = comp_df.loc[date]
        bc         = bench_comp.loc[date] if date in bench_comp.index else np.nan

        # ---- EXIT: stop_loss / time_exit / rank > k ----
        for sym in list(positions.keys()):
            r        = today_rank.get(sym, np.nan)
            cur_px   = float(prices_dict[sym].loc[date]) if date in prices_dict[sym].index else np.nan
            reason   = _exit_reason(
                positions[sym], cur_px if not pd.isna(cur_px) else positions[sym]["entry_px"],
                r, k, date, stop_loss, max_hold_days,
            )
            if reason is not None:
                if next_date in prices_dict[sym].index:
                    px_sell  = float(prices_dict[sym].loc[next_date]) * (1.0 - cost.slippage_rate)
                    proceeds = positions[sym]["qty"] * px_sell
                    comm     = proceeds * cost.commission_rate
                    cash    += proceeds - comm
                    exit_ret = px_sell / positions[sym]["entry_px"] - 1.0
                    positions[sym]["_exit_reason"] = reason
                    positions[sym]["_exit_ret"]    = exit_ret
                del positions[sym]
                n_trades += 1

        # ---- ENTRY: fill slots up to k ----
        open_slots = k - len(positions)
        if open_slots <= 0:
            continue

        # Market regime filter: TOPIX > MA200 のときのみ新規エントリー許可
        if ma200 is not None:
            bench_close = bench_prices.loc[date] if date in bench_prices.index else np.nan
            ma200_val   = ma200.loc[date]         if date in ma200.index       else np.nan
            if pd.isna(bench_close) or pd.isna(ma200_val) or bench_close <= ma200_val:
                continue   # 下降レジーム → エントリーしない（exitは上で済み）

        # top-k 候補（保有済み除外、ランク昇順）
        cands = today_rank[today_rank <= k].sort_values()
        cands = cands[~cands.index.isin(positions)]

        # RSモメンタムフィルター: composite_return > benchmark
        if momentum_filter and not pd.isna(bc):
            valid = today_comp[cands.index] > bc
            cands = cands[valid]

        for sym in cands.index:
            if len(positions) >= k:
                break
            if next_date not in prices_dict[sym].index:
                continue

            px_buy  = float(prices_dict[sym].loc[next_date]) * (1.0 + cost.slippage_rate)
            # equal-weight: cash を残スロット数で割る（capital/k が上限）
            alloc   = min(cash / max(1, open_slots), capital / k)
            qty     = int(alloc // (px_buy * 100)) * 100   # 単元株（100株）
            if qty <= 0:
                continue
            buy_cost = qty * px_buy * (1.0 + cost.commission_rate)
            if buy_cost > cash:
                continue

            cash -= buy_cost
            positions[sym] = {
                "qty":        qty,
                "cost_basis": buy_cost,
                "entry_px":   px_buy,
                "entry_date": next_date,
            }
            n_trades += 1
            open_slots -= 1

    # ---- 結果集計 ----
    eq        = pd.Series(eq_vals, index=all_dates)
    np_series = pd.Series(n_pos,   index=all_dates)

    ret  = eq.pct_change().dropna()
    sp   = float(ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else 0.0
    yrs  = len(eq) / 252.0
    cg   = float((eq.iloc[-1] / eq.iloc[0]) ** (1.0 / yrs) - 1) * 100
    roll_max = eq.expanding().max()
    mdd  = float((eq / roll_max - 1).min()) * 100
    aexp = float((np_series > 0).mean())
    eff  = cg / aexp if aexp > 0 else 0.0
    calmar = cg / abs(mdd) if mdd < 0 else float("inf")

    return {
        "equity":      eq,
        "n_positions": np_series,
        "n_trades":    n_trades,
        "sharpe":      sp,
        "cagr":        cg,
        "maxdd":       mdd,
        "avg_exp":     aexp,
        "efficiency":  eff,
        "calmar":      calmar,
    }


def _slice_metrics(res: dict, start: str, end: str) -> dict:
    """equity を start〜end に切り出してメトリクスを再計算"""
    eq  = res["equity"]
    nps = res["n_positions"]

    eq  = eq[(eq.index >= start) & (eq.index <= end)]
    nps = nps[(nps.index >= start) & (nps.index <= end)]

    if len(eq) < 2:
        return {"sharpe": 0.0, "cagr": 0.0, "maxdd": 0.0, "avg_exp": 0.0,
                "efficiency": 0.0, "calmar": 0.0}

    ret  = eq.pct_change().dropna()
    sp   = float(ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else 0.0
    yrs  = len(eq) / 252.0
    cg   = float((eq.iloc[-1] / eq.iloc[0]) ** (1.0 / yrs) - 1) * 100
    mdd  = float((eq / eq.expanding().max() - 1).min()) * 100
    aexp = float((nps > 0).mean())
    eff  = cg / aexp if aexp > 0 else 0.0
    calmar = cg / abs(mdd) if mdd < 0 else float("inf")

    return {"sharpe": sp, "cagr": cg, "maxdd": mdd,
            "avg_exp": aexp, "efficiency": eff, "calmar": calmar}


# ================================================================
# MAIN
# ================================================================
def main():
    print("=" * 72)
    print("  backtest/topk_rotation.py — Top-k ローテーション バックテスト")
    print(f"  IS : {IS_START} 〜 {IS_END}")
    print(f"  OOS: {OOS_START} 〜 {OOS_END}")
    print(f"  テストケース: k=2, k=3, k=4, k=5, k=3+RSフィルター")
    print("=" * 72)

    # ---- データ取得 ----
    print(f"\n[1/3] ユニバースデータ取得...")
    with open(UNIVERSE_JSON, encoding="utf-8") as f:
        d = json.load(f)
    tickers: dict[str, str] = d["symbols"]

    univ = download_universe(tickers, start=IS_START, end=FULL_END, verbose=False)
    prices_dict = {sym: info["df"]["Close"] for sym, info in univ.items()}
    print(f"  取得完了: {len(prices_dict)} 銘柄")

    # ベンチマーク（TOPIX ETF）
    print(f"  ベンチマーク取得 ({BENCH_TICKER})...")
    raw_bench = yf.download(BENCH_TICKER, start=IS_START, end=FULL_END, progress=False)
    if isinstance(raw_bench.columns, pd.MultiIndex):
        raw_bench = raw_bench.droplevel(1, axis=1)
    bench_prices = raw_bench["Close"].dropna()
    bench_comp   = calc_composite_return(bench_prices)
    print(f"  ベンチマーク: {len(bench_prices)} 日")

    # ---- バックテスト実行 ----
    print(f"\n[2/3] バックテスト実行...")

    # ケース定義: (label, k, stop_loss, max_hold_days)
    # k=4 基準モデル + stop-loss / time-exit の組み合わせ比較
    all_cases = [
        ("k=4_base",     4, None,  None),
        ("k=4_sl15",     4, -0.15, None),
        ("k=4_sl12",     4, -0.12, None),
        ("k=4_sl15_h60", 4, -0.15, 60  ),
        ("k=4_sl12_h60", 4, -0.12, 60  ),
    ]

    results = {}
    stop_stats = {}

    for label, k, sl, mh in all_cases:
        sl_str = f"SL={sl*100:.0f}%" if sl else "SL=なし"
        mh_str = f" H={mh}d"         if mh else ""
        print(f"  {label} ({sl_str}{mh_str}) 実行中...")
        r = run_topk(
            prices_dict, bench_comp,
            k=k, stop_loss=sl, max_hold_days=mh, bench_prices=bench_prices,
        )
        results[label] = r
        is_s = _slice_metrics(r, IS_START, IS_END)["sharpe"]
        is_d = _slice_metrics(r, IS_START, IS_END)["maxdd"]
        print(f"    IS  Sharpe={is_s:.3f}  MaxDD={is_d:.1f}%")

    # ---- 結果表示 ----
    print(f"\n[3/3] 結果サマリー")
    print("=" * 80)
    hdr = f"  {'ケース':<16} {'期間':<5} {'Sharpe':>7} {'CAGR%':>7} {'MaxDD%':>8} {'avg_exp':>8} {'Calmar':>7}"
    print(hdr)
    print("  " + "-" * 66)

    for label, k, sl, mh in all_cases:
        r = results[label]
        for period, start, end in [("IS", IS_START, IS_END), ("OOS", OOS_START, OOS_END)]:
            m = _slice_metrics(r, start, end)
            print(
                f"  {label:<16} {period:<5} "
                f"{m['sharpe']:>7.3f} {m['cagr']:>7.2f} "
                f"{m['maxdd']:>8.2f} {m['avg_exp']:>8.3f} "
                f"{m['calmar']:>7.3f}"
            )
        print()

    # ---- DD削減サマリー ----
    print("  ▼ IS MaxDD 比較（Phase 1 基準: -20%以内）")
    base_mdd = _slice_metrics(results["k=4_base"], IS_START, IS_END)["maxdd"]
    for label, k, sl, mh in all_cases:
        m   = _slice_metrics(results[label], IS_START, IS_END)
        imp = m["maxdd"] - base_mdd
        ok  = "✅" if m["maxdd"] > -20.0 else ("⚠" if m["maxdd"] > -25.0 else "❌")
        print(f"  {label:<16}: MaxDD={m['maxdd']:>7.2f}%  改善={imp:>+6.2f}%pt  Sharpe={m['sharpe']:.3f}  {ok}")
    print("=" * 80)

    # ---- チャート ----
    os.makedirs(OUTPUT, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    colors = ["#aaaaaa", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for (label, k, sl, mh), color in zip(all_cases, colors):
        eq = results[label]["equity"]
        eq_is  = eq[eq.index <= IS_END]
        eq_oos = eq[eq.index >= OOS_START]
        norm_is  = eq_is  / eq_is.iloc[0]
        norm_oos = eq_oos / eq_oos.iloc[0]
        axes[0].plot(eq_is.index,  norm_is,  color=color, label=label, linewidth=1.5)
        axes[1].plot(eq_oos.index, norm_oos, color=color, label=label, linewidth=1.5)

    axes[0].set_title("IS (2018-2024) — 資産推移（正規化）")
    axes[1].set_title("OOS (2025) — 資産推移（正規化）")
    for ax in axes:
        ax.legend(loc="upper left", fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("正規化資産")

    plt.tight_layout()
    chart_path = os.path.join(OUTPUT, "topk_rotation.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  チャート保存: {chart_path}")

    # ---- JSON保存 ----
    import datetime
    today = datetime.date.today().isoformat()
    out = {}
    for label, k, sl, mh in all_cases:
        out[label] = {
            "IS":  _slice_metrics(results[label], IS_START, IS_END),
            "OOS": _slice_metrics(results[label], OOS_START, OOS_END),
        }
    json_path = f"results/topk_rotation_{today}.json"
    os.makedirs("results", exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"  JSON保存: {json_path}")


if __name__ == "__main__":
    main()
