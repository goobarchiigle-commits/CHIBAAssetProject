"""
backtest/cross_validate.py
===========================
3重クロス検証スクリプト

【目的】
  同じフジコ法ロジックを3つの独立した実装で検証し、
  結果がすべて一致するかを確認する。

  全て一致  → 実装は正しい（ルックアヘッドなし）
  乖離あり  → どこかにバグ or ルックアヘッドがある

【実装】
  A: 自作 BacktestEngine (bar-by-bar, event-driven) ← 既存コード
  B: backtesting.py ライブラリ (第三者フレームワーク)
  C: pandas vectorized (シフトで先読みを明示的に防止)

【実行】
  cd C:/Users/owner/.gemini/antigravity/scratch/asset_simulation
  python -m backtest.cross_validate
  python -m backtest.cross_validate --symbol 8035.T
"""

from __future__ import annotations
import sys
import os
import argparse
import warnings

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backtest.engine import BacktestEngine, TradeCost, BacktestResult
from backtest.rsr import (
    calc_rsr_vs_benchmark,
    calc_rsr_momentum,
    calc_sepa,
    calc_composite_return,
)
from backtest.fujiko_strategy import FujikoStrategy


# ================================================================== #
# 設定
# ================================================================== #
DEFAULT_SYMBOL    = "1925.T"
BENCHMARK_TICKER  = "1306.T"
START             = "2018-01-01"
END               = "2024-12-31"
CAPITAL           = 2_000_000
COST              = TradeCost()

# フジコ法パラメータ（共通）
PARAMS = dict(
    min_sepa         = 6,
    min_rsr          = 70.0,
    mom_period       = 21,
    turtle_entry     = 20,
    turtle_exit      = 10,
    use_turtle_entry = True,
)


# ================================================================== #
# 共通：インジケーター事前計算
# ================================================================== #
def precompute_indicators(df: pd.DataFrame, bench_close: pd.Series) -> pd.DataFrame:
    """
    全インジケーターをデータフレーム全体で一括計算する。

    ※ ここでは計算を前日終値ベースにするため shift(1) を適用。
      各指標は「i日時点で知りうる値」になる。

    Returns:
        元のdf に以下の列を追加したDataFrame:
          rsr, rsr_mom, rsr_mom_prev, sepa_score,
          turtle_high (20日), turtle_low (10日)
    """
    out = df.copy()
    close = df["Close"]

    # --- RSR（ベンチマーク比較） ---
    bench_aligned = bench_close.reindex(df.index).ffill()
    rsr = calc_rsr_vs_benchmark(close, bench_aligned)
    out["rsr"] = rsr

    # --- RSRモメンタム ---
    mom = calc_rsr_momentum(rsr, period=PARAMS["mom_period"])
    out["rsr_mom"]      = mom
    out["rsr_mom_prev"] = mom.shift(1)   # 前日モメンタム

    # --- SEPA 8条件スコア ---
    sepa = calc_sepa(df, rsr)
    out["sepa_score"] = sepa["sepa_score"]

    # --- タートルズS1 ---
    # 「前日まで」の高値・安値なので shift(1) + rolling
    out["turtle_high"] = close.shift(1).rolling(PARAMS["turtle_entry"]).max()
    out["turtle_low"]  = close.shift(1).rolling(PARAMS["turtle_exit"]).min()

    return out


# ================================================================== #
# シグナル生成ロジック（1行 = 1日の判定、先読みなし前提）
# ================================================================== #
def compute_signal_series(ind: pd.DataFrame) -> pd.Series:
    """
    インジケーターDataFrame から日次シグナルを生成する。
    戻り値: +1(買い) / -1(売り) / 0(なし)

    ルール:
      売り優先。以下のどちらかで売り:
        ① RSRモメンタム < 0 かつ 下降中
        ② 終値 < turtle_low

      買い条件（全部満たすこと）:
        ① sepa_score >= min_sepa
        ② rsr >= min_rsr
        ③ RSRモメンタム > 0 かつ 上昇中
        ④ 終値 > turtle_high（use_turtle_entry=True の場合）
    """
    # 必要最低データ行数チェック
    min_bars = 252 + PARAMS["mom_period"] + 2
    signals = pd.Series(0, index=ind.index, dtype=int)

    valid = (
        ind["rsr"].notna() &
        ind["rsr_mom"].notna() &
        ind["rsr_mom_prev"].notna() &
        ind["sepa_score"].notna() &
        ind["turtle_high"].notna() &
        ind["turtle_low"].notna()
    )

    c = ind[valid].copy()

    sell1 = (c["rsr_mom"] < 0) & (c["rsr_mom"] < c["rsr_mom_prev"])
    sell2 = c["Close"] < c["turtle_low"]
    sell  = sell1 | sell2

    buy_sepa   = c["sepa_score"] >= PARAMS["min_sepa"]
    buy_rsr    = c["rsr"] >= PARAMS["min_rsr"]
    buy_mom    = (c["rsr_mom"] > 0) & (c["rsr_mom"] > c["rsr_mom_prev"])
    buy_turtle = c["Close"] > c["turtle_high"]   # use_turtle_entry=True
    buy        = buy_sepa & buy_rsr & buy_mom & buy_turtle

    sig = pd.Series(0, index=c.index, dtype=int)
    sig[buy]  = 1
    sig[sell] = -1   # 売り優先で上書き

    signals.update(sig)
    return signals


# ================================================================== #
# 実装 A: 自作 BacktestEngine
# ================================================================== #
def run_impl_a(df: pd.DataFrame, bench_close: pd.Series) -> BacktestResult:
    """自作エンジン（既存コード）で実行する。"""
    strategy = FujikoStrategy(
        benchmark_prices = bench_close,
        **PARAMS,
    )
    engine = BacktestEngine(
        prices  = df,
        strategy= strategy,
        capital = CAPITAL,
        cost    = COST,
        symbol  = "ImplA",
    )
    return engine.run()


# ================================================================== #
# 実装 B: backtesting.py ライブラリ
# ================================================================== #
def run_impl_b(df: pd.DataFrame, ind: pd.DataFrame) -> dict:
    """
    backtesting.py を使った独立実装。

    backtesting.py は:
      - next() は毎バー呼ばれる
      - self.data は現在バーまでのスライス（未来は見えない）
      - self.I() で宣言したインジケーターも同様
      - デフォルト: シグナル当日 → 翌日始値で約定
    """
    from backtesting import Backtest, Strategy as BTStrategy

    # backtesting.py は Column名を大文字で期待する
    bt_df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

    # インジケーター配列を合わせる（backtesting.pyは numpy 配列か pd.Series）
    rsr_arr      = ind["rsr"].values.astype(float)
    mom_arr      = ind["rsr_mom"].values.astype(float)
    mom_prev_arr = ind["rsr_mom_prev"].values.astype(float)
    sepa_arr     = ind["sepa_score"].values.astype(float)
    t_high_arr   = ind["turtle_high"].values.astype(float)
    t_low_arr    = ind["turtle_low"].values.astype(float)
    min_bars     = 252 + PARAMS["mom_period"] + 2

    class FujikoB(BTStrategy):
        def init(self):
            # self.I() は「全バーで計算済み」の配列をラップする
            # → next() でアクセスする時は現在バーまでのスライスになる
            self.rsr      = self.I(lambda: rsr_arr,      name="RSR")
            self.mom      = self.I(lambda: mom_arr,      name="Mom")
            self.mom_prev = self.I(lambda: mom_prev_arr, name="MomPrev")
            self.sepa     = self.I(lambda: sepa_arr,     name="SEPA")
            self.t_high   = self.I(lambda: t_high_arr,   name="THigh")
            self.t_low    = self.I(lambda: t_low_arr,    name="TLow")

        def next(self):
            # バー数が不十分なら何もしない
            if len(self.data) < min_bars:
                return

            rsr      = self.rsr[-1]
            mom      = self.mom[-1]
            mom_prev = self.mom_prev[-1]
            sepa     = self.sepa[-1]
            t_high   = self.t_high[-1]
            t_low    = self.t_low[-1]
            close    = self.data.Close[-1]

            if np.isnan(rsr) or np.isnan(mom) or np.isnan(t_high):
                return

            # --- 売り判定（ポジションあるときのみ）---
            if self.position:
                sell1 = (mom < 0) and (mom < mom_prev)
                sell2 = close < t_low
                if sell1 or sell2:
                    self.position.close()
                    return

            # --- 買い判定（ノーポジのときのみ）---
            if not self.position:
                buy = (
                    sepa >= PARAMS["min_sepa"] and
                    rsr  >= PARAMS["min_rsr"]  and
                    mom  >  0                  and
                    mom  >  mom_prev           and
                    close > t_high
                )
                if buy:
                    # 資本の95%でエントリー（残り5%は手数料バッファ）
                    self.buy(size=0.95)

    bt = Backtest(
        bt_df,
        FujikoB,
        cash      = CAPITAL,
        commission= COST.commission_rate + COST.slippage_rate,
        trade_on_close = False,   # 翌日始値で約定（自作エンジンと同条件）
        exclusive_orders = True,
    )
    stats = bt.run()
    return {
        "total_return"  : float(stats["Return [%]"]) / 100,
        "sharpe_ratio"  : float(stats["Sharpe Ratio"]) if not np.isnan(stats["Sharpe Ratio"]) else 0.0,
        "max_drawdown"  : float(stats["Max. Drawdown [%]"]) / 100,
        "num_trades"    : int(stats["# Trades"]),
        "win_rate"      : float(stats["Win Rate [%]"]) / 100 if stats["# Trades"] > 0 else 0.0,
        "equity_final"  : float(stats["Equity Final [$]"]),
        "_stats"        : stats,
        "_bt"           : bt,
    }


# ================================================================== #
# 実装 C: pandas vectorized
# ================================================================== #
def run_impl_c(df: pd.DataFrame, ind: pd.DataFrame) -> dict:
    """
    純粋な pandas 行列演算で実装する。

    先読み防止:
      シグナルは i 日の確定データで生成 → i+1 日の始値で約定。
      signals.shift(1) で1日ずらして実際の執行に対応。
    """
    signals = compute_signal_series(ind)

    # 翌日始値で約定するため1日シフト
    exec_signals = signals.shift(1).fillna(0).astype(int)

    open_prices = df["Open"]
    close_prices = df["Close"]

    cash     = float(CAPITAL)
    holding  = 0
    avg_cost = 0.0

    equity = []
    trades = []

    for i, (date, sig) in enumerate(exec_signals.items()):
        row = df.loc[date]

        if sig == 1 and holding == 0 and cash > 0:
            raw_price = open_prices.loc[date]
            exec_p    = raw_price * (1 + COST.slippage_rate)
            qty       = int(cash // (exec_p * 100)) * 100
            if qty > 0:
                tv         = qty * exec_p
                commission = max(tv * COST.commission_rate, COST.min_commission)
                if tv + commission <= cash:
                    cash    -= tv + commission
                    holding  = qty
                    avg_cost = exec_p
                    trades.append({"date": date, "side": "BUY", "price": exec_p, "pnl": 0.0})

        elif sig == -1 and holding > 0:
            raw_price  = open_prices.loc[date]
            exec_p     = raw_price * (1 - COST.slippage_rate)
            tv         = holding * exec_p
            commission = max(tv * COST.commission_rate, COST.min_commission)
            pnl        = (exec_p - avg_cost) * holding - commission
            cash      += tv - commission
            trades.append({"date": date, "side": "SELL", "price": exec_p, "pnl": pnl})
            holding  = 0
            avg_cost = 0.0

        equity.append(cash + holding * close_prices.loc[date])

    equity_series = pd.Series(equity, index=exec_signals.index)

    daily_ret = equity_series.pct_change().dropna()
    sharpe    = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0.0

    peak = equity_series.cummax()
    mdd  = float(((equity_series - peak) / peak).min())

    sell_trades = [t for t in trades if t["side"] == "SELL"]
    num_trades  = len(sell_trades)
    win_rate    = sum(1 for t in sell_trades if t["pnl"] > 0) / num_trades if num_trades > 0 else 0.0
    total_ret   = equity_series.iloc[-1] / CAPITAL - 1.0

    return {
        "total_return" : total_ret,
        "sharpe_ratio" : sharpe,
        "max_drawdown" : mdd,
        "num_trades"   : num_trades,
        "win_rate"     : win_rate,
        "equity_final" : equity_series.iloc[-1],
        "_equity"      : equity_series,
        "_trades"      : trades,
    }


# ================================================================== #
# 比較レポート出力
# ================================================================== #
def print_comparison(
    symbol: str,
    res_a: BacktestResult,
    res_b: dict,
    res_c: dict,
) -> None:
    """3実装の結果を並べて表示し、乖離を診断する。"""

    print("\n" + "=" * 65)
    print(f"  3重クロス検証レポート  /  銘柄: {symbol}  /  {START}〜{END}")
    print("=" * 65)
    print(f"{'指標':<22} {'A: 自作エンジン':>14} {'B: backtest.py':>14} {'C: vectorized':>14}")
    print("-" * 65)

    rows = [
        ("総リターン (%)",    res_a.total_return*100,  res_b["total_return"]*100,  res_c["total_return"]*100),
        ("シャープ比",        res_a.sharpe_ratio,       res_b["sharpe_ratio"],       res_c["sharpe_ratio"]),
        ("最大DD (%)",        res_a.max_drawdown*100,   res_b["max_drawdown"]*100,   res_c["max_drawdown"]*100),
        ("取引回数",          res_a.num_trades,          res_b["num_trades"],          res_c["num_trades"]),
        ("勝率 (%)",          res_a.win_rate*100,        res_b["win_rate"]*100,        res_c["win_rate"]*100),
        ("最終資産 (万円)",   res_a.equity_curve.iloc[-1]/10000,
                              res_b["equity_final"]/10000,
                              res_c["equity_final"]/10000),
    ]

    for label, va, vb, vc in rows:
        if isinstance(va, int):
            print(f"{label:<22} {va:>14d} {vb:>14d} {vc:>14d}")
        else:
            print(f"{label:<22} {va:>14.2f} {vb:>14.2f} {vc:>14.2f}")

    print("=" * 65)

    # ---- 乖離診断 ----
    print("\n【乖離診断】")

    def check(name, a, b, c, tol=0.05):
        diff_ab = abs(a - b)
        diff_ac = abs(a - c)
        diff_bc = abs(b - c)
        max_diff = max(diff_ab, diff_ac, diff_bc)
        if max_diff <= tol:
            print(f"  ✅ {name}: 3実装一致 (最大差 {max_diff:.4f})")
        else:
            print(f"  ❌ {name}: 乖離あり!  A={a:.4f}  B={b:.4f}  C={c:.4f}  "
                  f"(最大差 {max_diff:.4f})")

    check("総リターン", res_a.total_return, res_b["total_return"], res_c["total_return"], tol=0.02)
    check("シャープ比", res_a.sharpe_ratio, res_b["sharpe_ratio"], res_c["sharpe_ratio"], tol=0.10)
    check("最大DD",     res_a.max_drawdown, res_b["max_drawdown"], res_c["max_drawdown"], tol=0.02)

    n_a, n_b, n_c = res_a.num_trades, res_b["num_trades"], res_c["num_trades"]
    if n_a == n_b == n_c:
        print(f"  ✅ 取引回数: 3実装一致 ({n_a}回)")
    else:
        print(f"  ❌ 取引回数: 不一致!  A={n_a}  B={n_b}  C={n_c}")
        if n_a != n_c:
            print("     → A vs C 不一致: 自作エンジンと vectorized で取引タイミングがずれている")
            print("       原因候補: shift(1) の有無 / turtle_high の計算式")
        if n_a != n_b:
            print("     → A vs B 不一致: backtesting.py のインジケーター渡し方を確認")

    # ---- Phase 1 判定 ----
    print("\n【Phase 1 達成判定（自作エンジン基準）】")
    sharpe_ok = res_a.sharpe_ratio > 0.5
    dd_ok     = res_a.max_drawdown > -0.20
    print(f"  シャープ比 {res_a.sharpe_ratio:.3f} > 0.5 : {'✅' if sharpe_ok else '❌'}")
    print(f"  最大DD     {res_a.max_drawdown*100:.1f}% > -20%: {'✅' if dd_ok else '❌'}")
    if sharpe_ok and dd_ok:
        print("  → Phase 1 達成")
    else:
        print("  → Phase 1 未達")

    # ---- 取引数の統計的有意性 ----
    n = res_a.num_trades
    print(f"\n【統計的有意性】")
    if n < 5:
        print(f"  ⚠️  取引回数 {n}回 → 統計的に無意味（CLAUDE.md基準: 5回未満は不合格）")
    elif n < 30:
        print(f"  ⚠️  取引回数 {n}回 → Chanの基準30回未満。過学習リスクあり")
    else:
        print(f"  ✅ 取引回数 {n}回 → 統計的有意性あり")


# ================================================================== #
# グラフ出力
# ================================================================== #
def plot_comparison(
    symbol: str,
    df: pd.DataFrame,
    res_a: BacktestResult,
    res_c: dict,
    out_path: str = "data/cross_validate.png",
) -> None:
    plt.rcParams["font.family"] = "MS Gothic"
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    eq_a = res_a.equity_curve / 10_000
    eq_c = res_c["_equity"] / 10_000
    close_norm = df["Close"] / df["Close"].iloc[0] * CAPITAL / 10_000

    # ---- 上段: 資産推移 ----
    ax = axes[0]
    ax.plot(eq_a.index, eq_a.values, color="royalblue", lw=1.5, label=f"A: 自作エンジン {res_a.total_return*100:+.1f}%")
    ax.plot(eq_c.index, eq_c.values, color="tomato",    lw=1.2, ls="--", label=f"C: vectorized {res_c['total_return']*100:+.1f}%", alpha=0.8)
    ax.plot(close_norm.index, close_norm.values, color="gray", lw=0.8, ls=":", label=f"{symbol} 買持ち")
    ax.axhline(CAPITAL / 10_000, color="black", lw=0.6, ls="--")
    ax.set_ylabel("資産（万円）")
    ax.set_title(f"3重クロス検証 — {symbol}  ({START}〜{END})")
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"¥{x:.0f}万"))

    # ---- 中段: ドローダウン比較 ----
    ax = axes[1]
    peak_a = res_a.equity_curve.cummax()
    dd_a   = (res_a.equity_curve - peak_a) / peak_a * 100
    peak_c = res_c["_equity"].cummax()
    dd_c   = (res_c["_equity"] - peak_c) / peak_c * 100

    ax.fill_between(dd_a.index, dd_a.values, 0, color="royalblue", alpha=0.4, label="A: 自作エンジン")
    ax.plot(dd_c.index, dd_c.values, color="tomato", lw=0.8, ls="--", label="C: vectorized")
    ax.axhline(-20, color="red", lw=0.8, ls="-.", label="Phase1 DD限界 -20%")
    ax.set_ylabel("ドローダウン (%)")
    ax.legend(fontsize=9)

    # ---- 下段: A-C の差分（ルックアヘッド検出） ----
    ax = axes[2]
    diff = (res_a.equity_curve - res_c["_equity"]).reindex(res_a.equity_curve.index).fillna(0)
    ax.fill_between(diff.index, diff.values, 0,
                    where=diff >= 0, color="royalblue", alpha=0.5, label="A > C")
    ax.fill_between(diff.index, diff.values, 0,
                    where=diff <  0, color="tomato",    alpha=0.5, label="C > A")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("資産差額 A - C (¥)")
    ax.set_xlabel("日付")
    ax.legend(fontsize=9)
    note = ("差が一定方向に拡大し続ける → ルックアヘッドの可能性\n"
            "差がランダムに揺れる → 約定価格の端数誤差（許容範囲）")
    ax.set_title(f"A - C 差分  ({note})")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nグラフ保存: {out_path}")


# ================================================================== #
# パラメータ感度テスト（過学習チェック）
# ================================================================== #
def sensitivity_test(df: pd.DataFrame, bench_close: pd.Series) -> None:
    """
    min_rsr を ±10% 変化させてシャープ比がどう変わるかを確認。
    変化が大きすぎる → パラメータ依存が高い = 過学習リスク大。
    """
    print("\n【パラメータ感度テスト（min_rsr を変化させる）】")
    print(f"  基準値: min_rsr={PARAMS['min_rsr']:.0f}")
    print(f"  {'min_rsr':>8} {'取引':>6} {'Sharpe':>8} {'MaxDD':>8} {'変化(Sharpe)':>14}")
    print("  " + "-" * 50)

    base_sharpe = None
    for rsr_val in [55, 60, 65, 70, 75, 80, 85]:
        strategy = FujikoStrategy(
            benchmark_prices = bench_close,
            min_sepa         = PARAMS["min_sepa"],
            min_rsr          = float(rsr_val),
            mom_period       = PARAMS["mom_period"],
            turtle_entry     = PARAMS["turtle_entry"],
            turtle_exit      = PARAMS["turtle_exit"],
            use_turtle_entry = PARAMS["use_turtle_entry"],
        )
        eng = BacktestEngine(
            prices=df, strategy=strategy,
            capital=CAPITAL, cost=COST, symbol="sens"
        )
        r = eng.run()
        if rsr_val == int(PARAMS["min_rsr"]):
            base_sharpe = r.sharpe_ratio
        diff = (r.sharpe_ratio - base_sharpe) if base_sharpe is not None else 0.0
        mark = "← 基準" if rsr_val == int(PARAMS["min_rsr"]) else ""
        print(f"  {rsr_val:>8} {r.num_trades:>6} {r.sharpe_ratio:>8.3f} "
              f"{r.max_drawdown*100:>+7.1f}% {diff:>+14.3f} {mark}")

    if base_sharpe is not None:
        print(f"\n  判断基準: ±10の変化でシャープ比が±0.3以内 → 頑健")
        print(f"            ±0.3超 → パラメータ依存が高い（過学習リスク）")


# ================================================================== #
# メイン
# ================================================================== #
def main():
    parser = argparse.ArgumentParser(description="フジコ法 3重クロス検証")
    parser.add_argument("--symbol",      default=DEFAULT_SYMBOL, help="検証銘柄 (例: 8035.T)")
    parser.add_argument("--start",       default=START)
    parser.add_argument("--end",         default=END)
    parser.add_argument("--no-sensitivity", action="store_true", help="感度テストをスキップ")
    args = parser.parse_args()

    symbol = args.symbol
    print(f"\n{'='*65}")
    print(f"  フジコ法 3重クロス検証  |  {symbol}")
    print(f"{'='*65}")

    # --- データ取得 ---
    print(f"\n[1] データ取得中...")
    df = yf.download(symbol, start=args.start, end=args.end, progress=False)
    if df.empty:
        print(f"  エラー: {symbol} のデータを取得できませんでした")
        return
    df = df.droplevel(1, axis=1)

    bench_df = yf.download(BENCHMARK_TICKER, start=args.start, end=args.end, progress=False)
    bench_df = bench_df.droplevel(1, axis=1)
    bench_close = bench_df["Close"]

    print(f"  {symbol}: {len(df)}日")
    print(f"  ベンチマーク({BENCHMARK_TICKER}): {len(bench_df)}日")
    print(f"  パラメータ: {PARAMS}")

    # --- インジケーター計算 ---
    print(f"\n[2] インジケーター計算中...")
    ind = precompute_indicators(df, bench_close)

    # --- 3実装を実行 ---
    print(f"\n[3] 実装A: 自作BacktestEngine...")
    res_a = run_impl_a(df, bench_close)
    print(f"  完了: Sharpe={res_a.sharpe_ratio:.3f}  MaxDD={res_a.max_drawdown*100:.1f}%  "
          f"取引={res_a.num_trades}回")

    print(f"\n[4] 実装B: backtesting.py...")
    res_b = run_impl_b(df, ind)
    print(f"  完了: Sharpe={res_b['sharpe_ratio']:.3f}  MaxDD={res_b['max_drawdown']*100:.1f}%  "
          f"取引={res_b['num_trades']}回")

    print(f"\n[5] 実装C: pandas vectorized...")
    res_c = run_impl_c(df, ind)
    print(f"  完了: Sharpe={res_c['sharpe_ratio']:.3f}  MaxDD={res_c['max_drawdown']*100:.1f}%  "
          f"取引={res_c['num_trades']}回")

    # --- 比較レポート ---
    print_comparison(symbol, res_a, res_b, res_c)

    # --- グラフ ---
    print(f"\n[6] グラフ生成中...")
    out_path = f"data/cross_validate_{symbol.replace('.', '_')}.png"
    plot_comparison(symbol, df, res_a, res_c, out_path)

    # --- パラメータ感度テスト ---
    if not args.no_sensitivity:
        print(f"\n[7] パラメータ感度テスト...")
        sensitivity_test(df, bench_close)

    print(f"\n完了。")


if __name__ == "__main__":
    main()
