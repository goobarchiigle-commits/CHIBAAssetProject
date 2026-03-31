"""
backtest/portfolio_cross_validate.py
=====================================
フジコ法 ポートフォリオ全体 3重クロス検証

【目的】
  PortfolioEngine (V2設定) を3つの独立した方法で検証し、
  実装バグ・ルックアヘッドバイアス・過学習を検出する。

【3つの検証】
  A: 既存 PortfolioEngine  (本番コード、バー毎処理)
  B: 独立ベクトル化ポートフォリオ (pandas独自実装、PortfolioEngineと無関係)
  C: ウォークフォワード検証 (2年学習→1年テスト、過学習チェック)

【判断基準】
  A ≈ B      → 実装は正しい (差が 総リターン±2%, Sharpe±0.1, 取引数±10%)
  A ≠ B      → どこかにバグ or ロジック差異がある
  C (OOS) ≈ A (IS)  → 過学習なし (OOS Sharpe が IS の 70%以上)
  C (OOS) << A (IS) → 過学習の疑い

【実行】
  cd C:/Users/owner/.gemini/antigravity/scratch/asset_simulation
  python -m backtest.portfolio_cross_validate
  python -m backtest.portfolio_cross_validate --quick   # 27銘柄のみ・高速
"""

from __future__ import annotations
import sys, os, argparse, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "MS Gothic"

import yfinance as yf

from backtest.engine          import TradeCost
from backtest.portfolio_engine import PortfolioEngine, PortfolioResult
from backtest.fujiko_strategy  import FujikoStrategy
from backtest.mean_reversion_strategy import MeanReversionStrategy
from backtest.rsr              import calc_universe_rsr
from backtest.universe_builder import download_universe

# ================================================================== #
# V2 設定 (portfolio_v2.py の "V2" シナリオと同一)
# ================================================================== #
START   = "2018-01-01"
END     = "2024-12-31"
CAPITAL = 2_000_000
COST    = TradeCost()

V2_PARAMS = dict(
    max_positions     = 3,
    vol_target        = 0.0,     # 均等ウェイト
    max_single_weight = 0.25,
    use_idm           = False,
    min_sectors       = 1,
    max_dd_limit      = 0.15,
)

# セクター→戦略マッピング（portfolio_v2.py と同一）
SECTOR_STRATEGY = {
    "海運":"fujiko","機械":"fujiko","電機精密":"fujiko","商社":"fujiko",
    "電機":"fujiko","ゲーム":"fujiko","レジャー":"fujiko","食品":"fujiko",
    "ガス":"mean_rev","鉄鋼":"mean_rev","銀行":"mean_rev","保険":"mean_rev",
    "輸送機器":"mean_rev","化学":"mean_rev","小売":"mean_rev",
    "サービス":"dynamic","医薬品":"dynamic","不動産":"dynamic",
    "情報通信":"dynamic","陸運":"dynamic",
}
MR_PARAMS = dict(
    rsi_period=5, rsi_entry=25.0, rsi_exit=65.0,
    ma_long=200, stop_loss_pct=0.07, max_hold_days=10, knife_threshold=0.15,
)


# ================================================================== #
# ユーティリティ
# ================================================================== #
def _calmar(equity: pd.Series, initial: float) -> float:
    if equity.empty:
        return 0.0
    n_years = (equity.index[-1] - equity.index[0]).days / 365.25
    if n_years <= 0:
        return 0.0
    cagr = (equity.iloc[-1] / initial) ** (1 / n_years) - 1
    peak = equity.cummax()
    mdd  = float(((equity - peak) / peak).min())
    return cagr / abs(mdd) if mdd < 0 else float("inf")

def _sharpe(equity: pd.Series) -> float:
    r = equity.pct_change().dropna()
    return float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0

def _mdd(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float(((equity - peak) / peak).min())

def _cagr(equity: pd.Series, initial: float) -> float:
    n = (equity.index[-1] - equity.index[0]).days / 365.25
    return float((equity.iloc[-1] / initial) ** (1 / n) - 1) if n > 0 else 0.0

def _summary(label: str, equity: pd.Series, initial: float, n_trades: int) -> dict:
    return {
        "label"    : label,
        "cagr"     : _cagr(equity, initial),
        "sharpe"   : _sharpe(equity),
        "mdd"      : _mdd(equity),
        "calmar"   : _calmar(equity, initial),
        "n_trades" : n_trades,
        "final"    : equity.iloc[-1],
        "equity"   : equity,
    }


# ================================================================== #
# 戦略辞書を組み立て
# ================================================================== #
def build_strategies(universe: dict, rsr_uni: pd.DataFrame) -> dict:
    strats = {}
    for sym, info in universe.items():
        sector = info["sector"]
        rule   = SECTOR_STRATEGY.get(sector, "dynamic")
        rsr_s  = rsr_uni[sym] if sym in rsr_uni.columns else None
        if rule == "mean_rev":
            strats[sym] = MeanReversionStrategy(**MR_PARAMS)
        else:
            strats[sym] = FujikoStrategy(
                rsr_series       = rsr_s,
                min_sepa         = 6,
                min_rsr          = 70.0,
                mom_period       = 21,
                turtle_entry     = 20,
                turtle_exit      = 10,
                use_turtle_entry = True,
            )
    return strats


# ================================================================== #
# 実装 A : 既存 PortfolioEngine
# ================================================================== #
def run_impl_a(universe: dict, strats: dict) -> dict:
    """既存 PortfolioEngine (V2設定) で実行する。"""
    eng = PortfolioEngine(
        universe  = universe,
        strategy  = strats,
        capital   = CAPITAL,
        cost      = COST,
        **V2_PARAMS,
    )
    res = eng.run()

    n_trades = len(res.trades[res.trades["side"].str.startswith("SELL")]) if not res.trades.empty else 0
    return _summary("A: PortfolioEngine(V2)", res.equity_curve, CAPITAL, n_trades)


# ================================================================== #
# 実装 B : 独立ベクトル化ポートフォリオ
# ================================================================== #
def _precompute_all_signals(
    universe: dict, strats: dict, dates: pd.DatetimeIndex
) -> dict[str, pd.Series]:
    """
    全銘柄の全日付についてシグナルを事前計算する。

    PortfolioEngine の generate_signal(past) 呼び出しと完全に同じ条件:
      - past = df.loc[:date]  (今日の終値を含む)
      - 戻り値: +1 / -1 / 0
    """
    all_signals: dict[str, pd.Series] = {}
    for sym, info in universe.items():
        df_sym  = info["df"]
        strat   = strats[sym]
        sigs    = []
        sig_idx = []
        for date in dates:
            if date not in df_sym.index:
                continue
            past   = df_sym.loc[:date]
            signal = strat.generate_signal(past)
            sigs.append(signal)
            sig_idx.append(date)
        all_signals[sym] = pd.Series(sigs, index=pd.DatetimeIndex(sig_idx))
    return all_signals


def run_impl_b(universe: dict, strats: dict) -> dict:
    """
    独立ベクトル化ポートフォリオ。

    PortfolioEngine と同じルールを独自実装：
      - SELL: シグナル at t → 当日終値で決済 (PortfolioEngineと同じ)
      - BUY:  シグナル at t → 翌日始値で約定 (PortfolioEngineと同じ)
      - サーキットブレーカー: DD > 15% → 全決済、DD < 7.5% で解除
      - 最大3ポジション、均等ウェイト (capital/3 per slot)
      - セクター制約なし (min_sectors=1)
    """
    # 1. 全銘柄に共通する日付列
    dates: pd.DatetimeIndex | None = None
    for info in universe.values():
        idx   = info["df"].index
        dates = idx if dates is None else dates.intersection(idx)
    dates = dates.sort_values()

    # 2. シグナル事前計算
    print("    シグナル事前計算中 (全銘柄)...")
    all_signals = _precompute_all_signals(universe, strats, dates)

    # 3. 価格辞書
    close_d = {sym: info["df"]["Close"] for sym, info in universe.items()}
    open_d  = {sym: info["df"]["Open"]  for sym, info in universe.items()}

    # 4. ポートフォリオシミュレーション
    cash       = float(CAPITAL)
    positions  = {}          # {sym: {qty, entry_price, sector}}
    peak_eq    = float(CAPITAL)
    cb_active  = False
    equity_rec = []
    trades     = []
    alloc_per  = CAPITAL / V2_PARAMS["max_positions"]  # 均等配分 = 2,000,000/3

    sym_list = list(universe.keys())

    for i, date in enumerate(dates):

        # --- 時価評価 ---
        mkt = sum(
            pos["qty"] * float(close_d[sym].get(date, 0))
            for sym, pos in positions.items()
        )
        port_val = cash + mkt
        peak_eq  = max(peak_eq, port_val)
        curr_dd  = (port_val - peak_eq) / peak_eq

        # --- サーキットブレーカー ---
        if curr_dd < -V2_PARAMS["max_dd_limit"] and not cb_active:
            for sym in list(positions.keys()):
                price  = float(close_d[sym].get(date, 0))
                ep     = price * (1 - COST.slippage_rate)
                tv     = positions[sym]["qty"] * ep
                comm   = max(tv * COST.commission_rate, COST.min_commission)
                pnl    = (ep - positions[sym]["entry_price"]) * positions[sym]["qty"] - comm
                cash  += tv - comm
                trades.append({"date": date, "sym": sym, "side": "SELL_CB", "pnl": pnl})
            positions.clear()
            cb_active = True

        if cb_active and curr_dd > -V2_PARAMS["max_dd_limit"] * 0.5:
            cb_active = False

        # --- 売り判定（当日終値で決済） ---
        if not cb_active:
            for sym in list(positions.keys()):
                sig = all_signals.get(sym, pd.Series(dtype=int))
                if date in sig.index and sig.loc[date] == -1:
                    price = float(close_d[sym].get(date, 0))
                    ep    = price * (1 - COST.slippage_rate)
                    tv    = positions[sym]["qty"] * ep
                    comm  = max(tv * COST.commission_rate, COST.min_commission)
                    pnl   = (ep - positions[sym]["entry_price"]) * positions[sym]["qty"] - comm
                    cash += tv - comm
                    trades.append({"date": date, "sym": sym, "side": "SELL", "pnl": pnl})
                    del positions[sym]

        # --- 買い判定（翌日始値で約定） ---
        if not cb_active and i + 1 < len(dates):
            next_date = dates[i + 1]

            for sym in sym_list:
                if len(positions) >= V2_PARAMS["max_positions"]:
                    break
                if sym in positions:
                    continue

                sig = all_signals.get(sym, pd.Series(dtype=int))
                if date not in sig.index or sig.loc[date] != 1:
                    continue

                if next_date not in open_d[sym].index:
                    continue

                exec_p = float(open_d[sym].loc[next_date]) * (1 + COST.slippage_rate)
                # vol_target=0 時はPortfolioEngineと同様 capital/max_positions をそのまま使う
                # (max_single_weight上限はvol_target>0のときのみ適用されるため)
                alloc  = alloc_per
                qty    = int(alloc // (exec_p * 100)) * 100
                if qty <= 0:
                    continue

                tv   = qty * exec_p
                comm = max(tv * COST.commission_rate, COST.min_commission)
                if tv + comm > cash:
                    continue

                cash -= tv + comm
                positions[sym] = {
                    "qty": qty, "entry_price": exec_p,
                    "sector": universe[sym]["sector"]
                }
                trades.append({"date": next_date, "sym": sym, "side": "BUY", "pnl": 0.0})

        equity_rec.append({"date": date, "value": port_val})

    equity  = pd.Series(
        [r["value"] for r in equity_rec],
        index=pd.DatetimeIndex([r["date"] for r in equity_rec]),
    )
    n_sell = sum(1 for t in trades if t["side"] == "SELL")
    return _summary("B: VectorizedPortfolio", equity, CAPITAL, n_sell)


# ================================================================== #
# 実装 C : ウォークフォワード検証
# ================================================================== #
def run_impl_c_walkforward(universe: dict) -> list[dict]:
    """
    ウォークフォワード検証。

    ウィンドウ設定:
      - 学習期間: 2年 (in-sample)
      - テスト期間: 1年 (out-of-sample)
      - ステップ: 1年

    使用パラメータ: V2設定固定（パラメータ最適化なし）
    目的: 学習期間外でも成立するか確認

    判断基準:
      OOS Sharpe >= IS Sharpe × 0.7 → 過学習なし
      OOS Sharpe <  IS Sharpe × 0.5 → 過学習の疑い
    """
    windows = [
        ("2018-2019(IS)", "2020(OOS)", "2018-01-01", "2019-12-31", "2020-01-01", "2020-12-31"),
        ("2019-2020(IS)", "2021(OOS)", "2019-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
        ("2020-2021(IS)", "2022(OOS)", "2020-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
        ("2021-2022(IS)", "2023(OOS)", "2021-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
        ("2022-2023(IS)", "2024(OOS)", "2022-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    ]

    results = []
    for (is_label, oos_label, is_s, is_e, oos_s, oos_e) in windows:
        def _slice(univ: dict, start: str, end: str) -> dict:
            sliced = {}
            for sym, info in univ.items():
                df_slice = info["df"].loc[start:end]
                if len(df_slice) > 100:
                    sliced[sym] = {"df": df_slice, "sector": info["sector"]}
            return sliced

        univ_is  = _slice(universe, is_s, is_e)
        univ_oos = _slice(universe, oos_s, oos_e)

        if len(univ_is) < 5 or len(univ_oos) < 5:
            continue

        # IS期間のRSR
        prices_is  = {sym: info["df"]["Close"] for sym, info in univ_is.items()}
        rsr_is     = calc_universe_rsr(prices_is)
        strats_is  = build_strategies(univ_is, rsr_is)

        # OOS期間のRSR（IS+OOSで計算してOOS部分を使用 → ルークアヘッド防止）
        all_syms   = {**univ_is, **univ_oos}
        all_syms_v2 = {sym: {"df": universe[sym]["df"].loc[is_s:oos_e],
                              "sector": universe[sym]["sector"]}
                        for sym in all_syms if sym in universe}
        prices_full = {sym: info["df"]["Close"] for sym, info in all_syms_v2.items()}
        rsr_full    = calc_universe_rsr(prices_full)
        strats_oos  = build_strategies(univ_oos, rsr_full)

        # IS バックテスト
        res_is = PortfolioEngine(
            universe=univ_is, strategy=strats_is, capital=CAPITAL, cost=COST, **V2_PARAMS
        ).run()

        # OOS バックテスト
        res_oos = PortfolioEngine(
            universe=univ_oos, strategy=strats_oos, capital=CAPITAL, cost=COST, **V2_PARAMS
        ).run()

        n_is  = len(res_is.trades[res_is.trades["side"] == "SELL"]) if not res_is.trades.empty else 0
        n_oos = len(res_oos.trades[res_oos.trades["side"] == "SELL"]) if not res_oos.trades.empty else 0

        is_r  = _summary(f"IS {is_label}",   res_is.equity_curve,  CAPITAL, n_is)
        oos_r = _summary(f"OOS {oos_label}", res_oos.equity_curve, CAPITAL, n_oos)

        results.append({"is": is_r, "oos": oos_r})

    return results


# ================================================================== #
# 比較レポート出力
# ================================================================== #
def print_comparison(res_a: dict, res_b: dict) -> None:
    print("\n" + "=" * 68)
    print("  3重クロス検証  ポートフォリオ全体  A vs B 比較")
    print("=" * 68)
    print(f"{'指標':<22} {'A: PortfolioEngine':>20} {'B: Vectorized':>20}")
    print("-" * 68)

    rows = [
        ("CAGR (%)",       res_a["cagr"]*100,    res_b["cagr"]*100),
        ("Sharpe",         res_a["sharpe"],       res_b["sharpe"]),
        ("最大DD (%)",     res_a["mdd"]*100,      res_b["mdd"]*100),
        ("Calmar",         res_a["calmar"],        res_b["calmar"]),
        ("取引回数(売)",   float(res_a["n_trades"]), float(res_b["n_trades"])),
        ("最終資産 (万)",  res_a["final"]/10000,  res_b["final"]/10000),
    ]
    for label, va, vb in rows:
        print(f"{label:<22} {va:>20.2f} {vb:>20.2f}")
    print("=" * 68)

    # 乖離診断
    print("\n【乖離診断】")

    def chk(name, a, b, tol):
        diff = abs(a - b)
        if diff <= tol:
            print(f"  ✅ {name}: 一致 (差 {diff:.4f})")
        else:
            print(f"  ❌ {name}: 乖離! A={a:.4f} B={b:.4f} 差={diff:.4f}")

    chk("CAGR",     res_a["cagr"],    res_b["cagr"],    0.02)
    chk("Sharpe",   res_a["sharpe"],  res_b["sharpe"],  0.10)
    chk("MaxDD",    res_a["mdd"],     res_b["mdd"],     0.02)

    na, nb = res_a["n_trades"], res_b["n_trades"]
    diff_n = abs(na - nb)
    if diff_n <= max(2, na * 0.10):
        print(f"  ✅ 取引数: 一致 A={na} B={nb} (差 {diff_n})")
    else:
        print(f"  ❌ 取引数: 乖離! A={na} B={nb} (差 {diff_n})")
        print(f"     → 原因候補: SELL判定タイミング(当日close vs 翌日open)の差異")

    # PortfolioEngine の売り執行が「当日終値」という非標準ルール
    print("\n【注意: 実装上の非対称性】")
    print("  PortfolioEngine は SELL=当日終値 / BUY=翌日始値 で執行。")
    print("  BacktestEngine (単体) は SELL/BUY ともに翌日始値。")
    print("  この差異は意図的設計か要確認。")


def print_walkforward(wf_results: list[dict]) -> None:
    print("\n" + "=" * 68)
    print("  ウォークフォワード検証  (C) ―― 過学習チェック")
    print("=" * 68)
    print(f"{'期間':<22} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8} {'取引':>6} {'IS/OOS':>10}")
    print("-" * 68)

    is_sharpes  = []
    oos_sharpes = []

    for w in wf_results:
        ir = w["is"]
        or_ = w["oos"]
        ratio = or_["sharpe"] / ir["sharpe"] if ir["sharpe"] > 0.01 else float("nan")
        is_sharpes.append(ir["sharpe"])
        oos_sharpes.append(or_["sharpe"])

        lbl_is  = ir["label"].replace("IS ", "")
        lbl_oos = or_["label"].replace("OOS ", "")

        print(f"  IS  {lbl_is:<18} {ir['cagr']*100:>+7.1f}% {ir['sharpe']:>8.3f}"
              f" {ir['mdd']*100:>+7.1f}% {ir['n_trades']:>6}")
        print(f"  OOS {lbl_oos:<18} {or_['cagr']*100:>+7.1f}% {or_['sharpe']:>8.3f}"
              f" {or_['mdd']*100:>+7.1f}% {or_['n_trades']:>6}"
              f"  比={ratio:>5.2f}" if not np.isnan(ratio) else "")
        print()

    if is_sharpes and oos_sharpes:
        avg_is  = np.nanmean(is_sharpes)
        avg_oos = np.nanmean(oos_sharpes)
        ratio   = avg_oos / avg_is if avg_is > 0.01 else float("nan")
        print(f"  平均 IS  Sharpe: {avg_is:.3f}")
        print(f"  平均 OOS Sharpe: {avg_oos:.3f}")
        print(f"  OOS/IS 比率    : {ratio:.2f}" if not np.isnan(ratio) else "")
        print()
        if not np.isnan(ratio):
            if ratio >= 0.70:
                print("  ✅ OOS/IS >= 0.70 → 過学習なし（頑健性あり）")
            elif ratio >= 0.50:
                print("  ⚠️  OOS/IS 0.50〜0.70 → 軽微な過学習の可能性")
            else:
                print("  ❌ OOS/IS < 0.50 → 過学習の疑い強い")


# ================================================================== #
# グラフ出力
# ================================================================== #
def plot_results(res_a: dict, res_b: dict, wf_results: list[dict],
                 out_path: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 13))

    # ---- 上段: A vs B 資産推移 ----
    ax = axes[0]
    eq_a = res_a["equity"] / 10_000
    eq_b = res_b["equity"] / 10_000
    ax.plot(eq_a.index, eq_a.values, color="royalblue", lw=2.0,
            label=f"A PortfolioEngine  CAGR={res_a['cagr']*100:+.1f}%  Calmar={res_a['calmar']:.2f}")
    ax.plot(eq_b.index, eq_b.values, color="tomato", lw=1.5, ls="--",
            label=f"B Vectorized       CAGR={res_b['cagr']*100:+.1f}%  Calmar={res_b['calmar']:.2f}")
    ax.axhline(CAPITAL/10_000, color="gray", lw=0.8, ls="--")
    ax.set_ylabel("資産（万円）")
    ax.set_title("A vs B 資産推移比較")
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"¥{x:.0f}万"))

    # ---- 中段: A-B 差分（ルックアヘッド検出）----
    ax = axes[1]
    diff = (res_a["equity"] - res_b["equity"].reindex(res_a["equity"].index).ffill()).fillna(0)
    ax.fill_between(diff.index, diff/10_000, 0,
                    where=diff >= 0, color="royalblue", alpha=0.5, label="A > B")
    ax.fill_between(diff.index, diff/10_000, 0,
                    where=diff < 0,  color="tomato",    alpha=0.5, label="B > A")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("差額 A-B (万円)")
    ax.set_title("A-B 差分  | 一方向に拡大 → ルックアヘッド疑い / ランダム揺れ → 許容誤差")
    ax.legend(fontsize=9)

    # ---- 下段: ウォークフォワード IS vs OOS ----
    ax = axes[2]
    if wf_results:
        labels  = []
        is_s  = []
        oos_s = []
        for w in wf_results:
            yr = w["oos"]["label"].replace("OOS ", "")
            labels.append(yr)
            is_s.append(w["is"]["sharpe"])
            oos_s.append(w["oos"]["sharpe"])

        x = np.arange(len(labels))
        ax.bar(x - 0.2, is_s,  0.4, label="IS Sharpe",  color="steelblue", alpha=0.8)
        ax.bar(x + 0.2, oos_s, 0.4, label="OOS Sharpe", color="salmon",    alpha=0.8)
        ax.axhline(0, color="black", lw=0.5)
        ax.axhline(0.5, color="green", lw=0.8, ls="--", label="Phase1閾値 0.5")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Sharpe比")
        ax.set_title("ウォークフォワード  IS vs OOS Sharpe比（過学習チェック）")
        ax.legend(fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nグラフ保存: {out_path}")


# ================================================================== #
# メイン
# ================================================================== #
def main():
    parser = argparse.ArgumentParser(description="フジコ法 ポートフォリオ 3重クロス検証")
    parser.add_argument("--quick", action="store_true",
                        help="高速モード: dynamic_selection.csv の上位15銘柄のみ")
    parser.add_argument("--no-wf", action="store_true",
                        help="ウォークフォワードをスキップ（高速化）")
    args = parser.parse_args()

    print("=" * 68)
    print("  フジコ法 ポートフォリオ 3重クロス検証")
    print(f"  期間: {START} 〜 {END}  /  初期資本: ¥{CAPITAL:,}")
    print("=" * 68)

    # ---- ユニバース読み込み ----
    print("\n[1] ユニバース・データ取得中...")
    df_sel     = pd.read_csv("data/dynamic_selection.csv", encoding="utf-8-sig")
    mask       = (df_sel["sharpe"] > 0.30) & (df_sel["maxdd"].abs() < 0.30)
    tickers_27 = {r["symbol"]: r["sector"]
                  for _, r in df_sel[mask].iterrows()}

    if args.quick:
        # 上位15銘柄のみ（CAGRの高い順）
        top15 = df_sel[mask].nlargest(15, "cagr")
        tickers_use = {r["symbol"]: r["sector"] for _, r in top15.iterrows()}
        print(f"  高速モード: {len(tickers_use)}銘柄")
    else:
        tickers_use = tickers_27
        print(f"  通常モード: {len(tickers_use)}銘柄")

    universe = download_universe(tickers_use, start=START, end=END, verbose=False)
    print(f"  取得完了: {len(universe)}銘柄")

    # ---- RSR計算 ----
    print("\n[2] RSR計算中...")
    prices_all = {sym: info["df"]["Close"] for sym, info in universe.items()}
    rsr_uni    = calc_universe_rsr(prices_all)
    print(f"  完了: {rsr_uni.shape[1]}銘柄")

    # ---- 戦略辞書 ----
    strats = build_strategies(universe, rsr_uni)

    # ---- 実装 A: PortfolioEngine ----
    print("\n[3] 実装A: PortfolioEngine (V2設定)...")
    res_a = run_impl_a(universe, strats)
    print(f"  完了: CAGR={res_a['cagr']*100:+.1f}%  Sharpe={res_a['sharpe']:.3f}  "
          f"MaxDD={res_a['mdd']*100:.1f}%  Calmar={res_a['calmar']:.3f}  "
          f"取引={res_a['n_trades']}回")

    # ---- 実装 B: Vectorized ----
    print("\n[4] 実装B: 独立ベクトル化ポートフォリオ...")
    strats_b = build_strategies(universe, rsr_uni)   # 独立したインスタンス
    res_b = run_impl_b(universe, strats_b)
    print(f"  完了: CAGR={res_b['cagr']*100:+.1f}%  Sharpe={res_b['sharpe']:.3f}  "
          f"MaxDD={res_b['mdd']*100:.1f}%  Calmar={res_b['calmar']:.3f}  "
          f"取引={res_b['n_trades']}回")

    # ---- 比較レポート ----
    print_comparison(res_a, res_b)

    # ---- 実装 C: ウォークフォワード ----
    wf_results = []
    if not args.no_wf:
        print("\n[5] 実装C: ウォークフォワード検証 (5ウィンドウ)...")
        print("    ※ 時間がかかります（5分程度）...")
        wf_results = run_impl_c_walkforward(universe)
        print_walkforward(wf_results)
    else:
        print("\n[5] ウォークフォワードはスキップ (--no-wf)")

    # ---- グラフ ----
    print("\n[6] グラフ生成中...")
    plot_results(res_a, res_b, wf_results, "data/portfolio_cross_validate.png")

    # ---- 最終サマリー ----
    print("\n" + "=" * 68)
    print("  最終結果サマリー")
    print("=" * 68)
    print(f"{'':6} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8} {'Calmar':>8} {'取引':>6}")
    for r in [res_a, res_b]:
        print(f"  {r['label'][:30]:<30} {r['cagr']*100:>+7.1f}%"
              f" {r['sharpe']:>8.3f} {r['mdd']*100:>+7.1f}%"
              f" {r['calmar']:>8.3f} {r['n_trades']:>6}回")

    # Phase 1 基準チェック (A基準)
    print("\n【Phase 1 基準チェック (実装A)】")
    print(f"  Sharpe {res_a['sharpe']:.3f} > 0.5 : {'✅' if res_a['sharpe']>0.5 else '❌'}")
    print(f"  MaxDD  {res_a['mdd']*100:.1f}% > -20%: {'✅' if res_a['mdd']>-0.20 else '❌'}")
    print(f"  Calmar {res_a['calmar']:.3f} > 1.0 : {'✅' if res_a['calmar']>1.0 else '❌'}")
    calmar_v2_target = 2.656
    print(f"  Calmar {res_a['calmar']:.3f} ≈ 2.656(目標): "
          f"{'✅' if abs(res_a['calmar']-calmar_v2_target)<0.3 else '⚠ 乖離あり'}")

    print("\n完了。")


if __name__ == "__main__":
    main()
