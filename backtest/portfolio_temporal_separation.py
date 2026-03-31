"""
backtest/portfolio_temporal_separation.py
時間的分離バックテスト — 銘柄選択バイアスの定量化

【設計原則】
  Selection期間: 2015-01-01〜2017-12-31  ← フィルタはここだけ（eval期間を見ない）
  Eval期間:      2018-01-01〜2024-12-31  ← ポートフォリオ評価はここだけ
  閾値: Sharpe>0.3, MaxDD>-0.30 (BIASED版と同一・閾値の再最適化禁止)

【比較3シナリオ】
  BIASED     : 現行（2018-2024選定→2018-2024評価）  ← バイアスあり・基準線
  TEMPORAL   : 2015-2017選定→2018-2024評価           ← 時間的分離・バイアスなし
  TOP2_SECTOR: selection期間の出来高上位2社/セクター  ← パフォーマンス選定なし

【解釈ガイド】
  BIASED - TEMPORAL > 0.5 Sharpe → in-sample選択バイアスが支配的
  TEMPORAL Sharpe > 0.5          → 戦略に実在するアルファがある
  TEMPORAL Sharpe < 0.3          → 戦略の根本再設計が必要
"""

from __future__ import annotations
import os, sys, warnings
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

from backtest.universe_builder        import TOPIX100_TICKERS, download_universe
from backtest.rsr                     import calc_universe_rsr
from backtest.fujiko_strategy         import FujikoStrategy
from backtest.mean_reversion_strategy import MeanReversionStrategy
from backtest.portfolio_engine        import PortfolioEngine
from backtest.engine                  import BacktestEngine, TradeCost

OUTPUT = "C:/ai-trading/reports"
os.makedirs(OUTPUT, exist_ok=True)

# ------------------------------------------------------------------ #
# 期間定義（完全分離・変更禁止）
# ------------------------------------------------------------------ #
SEL_START  = "2015-01-01"   # Selection: ここでのみ銘柄を選ぶ
SEL_END    = "2017-12-31"
EVAL_START = "2018-01-01"   # Evaluation: ここでのみポートフォリオを評価する
EVAL_END   = "2024-12-31"

# フィルタ閾値（BIASED版と同一・再最適化禁止）
SHARPE_THRESH = 0.30
MAXDD_THRESH  = -0.30
MIN_SEL_DAYS  = 400   # selection期間の最低営業日数（約2年相当）

CAPITAL = 2_000_000

# ------------------------------------------------------------------ #
# セクター→戦略マッピング（事前情報のみ。バックテスト結果に依存しない）
# ------------------------------------------------------------------ #
SECTOR_STRATEGY: dict[str, str] = {
    "海運":     "fujiko",
    "機械":     "fujiko",
    "電機精密": "fujiko",
    "商社":     "fujiko",
    "電機":     "fujiko",
    "ゲーム":   "fujiko",
    "レジャー": "fujiko",
    "食品":     "fujiko",
    "情報通信": "fujiko",
    "サービス": "fujiko",
    "ガス":     "mean_rev",
    "鉄鋼":     "mean_rev",
    "銀行":     "mean_rev",
    "保険":     "mean_rev",
    "輸送機器": "mean_rev",
    "化学":     "mean_rev",
    "小売":     "mean_rev",
    "医薬品":   "mean_rev",
    "不動産":   "mean_rev",
    "陸運":     "mean_rev",
    "石油":     "mean_rev",
}

MR_PARAMS = dict(
    rsi_period=5, rsi_entry=25.0, rsi_exit=65.0,
    ma_long=200, stop_loss_pct=0.07, max_hold_days=10, knife_threshold=0.15,
)


# ------------------------------------------------------------------ #
# ヘルパー
# ------------------------------------------------------------------ #
def calmar(res) -> float:
    dd = abs(res.max_drawdown)
    return res.cagr / dd if dd > 0 else float("inf")


def metrics_row(res, label: str) -> dict:
    return {
        "シナリオ":       label,
        "CAGR%":         f"{res.cagr * 100:+.2f}",
        "MaxDD%":        f"{res.max_drawdown * 100:.2f}",
        "Calmar":        f"{calmar(res):.3f}",
        "Sharpe":        f"{res.sharpe_ratio:.3f}",
        "取引数":        res.n_trades,
        "勝率%":         f"{res.win_rate * 100:.1f}",
        "平均保有銘柄":  f"{res.n_positions.mean():.2f}",
        "最終資産(万)":  f"¥{res.equity_curve.iloc[-1] / 10_000:.0f}万",
        "_res":           res,
    }


def make_strategy(sym: str, sector: str, rsr_uni: pd.DataFrame, min_rsr: float = 70.0):
    """セクターマッピングに基づき戦略を生成。rsr_uniはeval期間のRSR。"""
    rsr_s = rsr_uni[sym] if sym in rsr_uni.columns else None
    rule  = SECTOR_STRATEGY.get(sector, "fujiko")
    if rule == "mean_rev":
        return MeanReversionStrategy(**MR_PARAMS)
    return FujikoStrategy(
        rsr_series=rsr_s, min_sepa=6, min_rsr=min_rsr,
        mom_period=21, turtle_entry=20, turtle_exit=20,
        use_turtle_entry=True,
    )


def screen_on_selection_period(
    universe_sel: dict, cost: TradeCost, rsr_sel: pd.DataFrame
) -> list[str]:
    """
    Selection期間データのみでシングル銘柄バックテストし合格銘柄を返す。
    ★ eval期間データは一切使わない。
    """
    passed = []
    print(f"\n  === Selection期間スクリーニング "
          f"({SEL_START}〜{SEL_END} / {len(universe_sel)}銘柄) ===")
    for sym, info in sorted(universe_sel.items()):
        df_s   = info["df"]
        sector = info["sector"]
        # selection期間のRSRで戦略生成（eval期間RSRを使わない）
        strat  = make_strategy(sym, sector, rsr_sel, min_rsr=70.0)
        eng    = BacktestEngine(
            prices=df_s, strategy=strat,
            capital=1_000_000, cost=cost, symbol=sym,
        )
        r   = eng.run()
        ok  = r.sharpe_ratio > SHARPE_THRESH and r.max_drawdown > MAXDD_THRESH
        mark = "✅" if ok else "✗ "
        print(f"  {mark} {sym:<8} ({sector:<8}) "
              f"Sharpe={r.sharpe_ratio:+.3f}  MaxDD={r.max_drawdown * 100:.1f}%  "
              f"取引={r.num_trades}")
        if ok:
            passed.append(sym)

    print(f"\n  合格: {len(passed)}/{len(universe_sel)}銘柄 "
          f"({', '.join(passed)})")
    return passed


def top2_per_sector_by_volume(universe_sel: dict) -> list[str]:
    """
    各セクターの出来高上位2銘柄を返す。
    ★ パフォーマンス指標を一切使わない純粋な事前情報。
    """
    sector_syms: dict[str, list[tuple[str, float]]] = {}
    for sym, info in universe_sel.items():
        sector  = info["sector"]
        avg_vol = float(info["df"]["Volume"].mean())
        sector_syms.setdefault(sector, []).append((sym, avg_vol))

    selected = []
    print(f"\n  === Top2/セクター選定（出来高上位） ===")
    for sector in sorted(sector_syms):
        sym_vols = sector_syms[sector]
        top2     = sorted(sym_vols, key=lambda x: x[1], reverse=True)[:2]
        names    = [s for s, _ in top2]
        vols     = [f"{v/1e6:.1f}M" for _, v in top2]
        print(f"  {sector:<10}: {names}  (出来高: {vols})")
        selected.extend(names)
    return selected


def run_portfolio(universe_eval: dict, strat_dict: dict) -> object:
    """ポートフォリオエンジン実行（V2相当パラメータ: 均等ウェイト）"""
    return PortfolioEngine(
        universe=universe_eval,
        strategy=strat_dict,
        capital=CAPITAL,
        max_dd_limit=0.15,
        min_sectors=1,
        max_positions=3,
        cost=TradeCost(),
        vol_target=0.0,          # 均等ウェイト（V2相当）
        max_single_weight=0.25,
        vol_window=20,
        use_idm=False,
        use_dd_scalar=False,
    ).run()


def yearly_table(res) -> pd.DataFrame:
    rows = []
    eq = res.equity_curve
    dd = res.dd_series
    for yr in range(2018, 2025):
        ye = eq[eq.index.year == yr]
        yd = dd[dd.index.year == yr]
        if ye.empty:
            continue
        ret = ye.iloc[-1] / ye.iloc[0] - 1
        mdd = float(yd.min())
        cal = ret / abs(mdd) if mdd < 0 else None
        rows.append({
            "年":        yr,
            "リターン%": round(ret * 100, 2),
            "MaxDD%":    round(mdd * 100, 2),
            "Calmar":    round(cal, 2) if cal else "∞",
        })
    return pd.DataFrame(rows)


# ================================================================
# MAIN
# ================================================================
def main():
    print("=" * 72)
    print("  portfolio_temporal_separation.py — バイアス定量化")
    print(f"  Selection : {SEL_START} 〜 {SEL_END}")
    print(f"  Evaluation: {EVAL_START} 〜 {EVAL_END}")
    print(f"  閾値: Sharpe>{SHARPE_THRESH}, MaxDD>{MAXDD_THRESH*100:.0f}%")
    print("=" * 72)

    cost = TradeCost()

    # ----------------------------------------------------------------
    # Step 1: Selection期間データ取得（全74銘柄）
    # ----------------------------------------------------------------
    print(f"\n[1/7] Selection期間データ取得 ({len(TOPIX100_TICKERS)}銘柄)...")
    univ_sel = download_universe(
        TOPIX100_TICKERS,
        start=SEL_START, end=SEL_END,
        min_days=MIN_SEL_DAYS,
        verbose=False,
    )
    print(f"  取得完了: {len(univ_sel)} 銘柄")

    # ----------------------------------------------------------------
    # Step 2: Selection期間のRSR計算
    #   ★ ここでeval期間のデータを混入させない
    # ----------------------------------------------------------------
    print("\n[2/7] Selection期間 RSR計算...")
    prices_sel = {sym: info["df"]["Close"] for sym, info in univ_sel.items()}
    rsr_sel    = calc_universe_rsr(prices_sel)
    print(f"  RSR計算完了: {rsr_sel.shape[1]} 銘柄")

    # ----------------------------------------------------------------
    # Step 3: 時間的分離スクリーニング（Selection期間のみ）
    # ----------------------------------------------------------------
    print("\n[3/7] 時間的分離スクリーニング...")
    temporal_syms = screen_on_selection_period(univ_sel, cost, rsr_sel)

    # ----------------------------------------------------------------
    # Step 4: Top2/セクター選定（出来高）
    # ----------------------------------------------------------------
    print("\n[4/7] Top2/セクター選定...")
    top2_syms = top2_per_sector_by_volume(univ_sel)
    print(f"  合計: {len(top2_syms)}銘柄")

    # ----------------------------------------------------------------
    # Step 5: BIASEDユニバース読み込み（既存 dynamic_selection.csv）
    #   ★ このCSVは2018-2024のバックテスト結果で作成済み（in-sample）
    # ----------------------------------------------------------------
    print("\n[5/7] BIASEDユニバース読み込み...")
    df_sel_csv   = pd.read_csv("data/dynamic_selection.csv", encoding="utf-8-sig")
    mask_biased  = (
        (df_sel_csv["sharpe"] > SHARPE_THRESH) &
        (df_sel_csv["maxdd"].abs() < abs(MAXDD_THRESH))
    )
    biased_tickers: dict[str, str] = {
        r["symbol"]: r["sector"]
        for _, r in df_sel_csv[mask_biased].iterrows()
    }
    print(f"  BIASEDユニバース: {len(biased_tickers)}銘柄")

    # ----------------------------------------------------------------
    # Step 6: Eval期間データ取得（3シナリオ分を一括）
    # ----------------------------------------------------------------
    all_eval_needed: dict[str, str] = {**biased_tickers}
    for sym in temporal_syms:
        if sym in TOPIX100_TICKERS:
            all_eval_needed.setdefault(sym, TOPIX100_TICKERS[sym])
    for sym in top2_syms:
        if sym in TOPIX100_TICKERS:
            all_eval_needed.setdefault(sym, TOPIX100_TICKERS[sym])

    print(f"\n[6/7] Eval期間データ取得 ({len(all_eval_needed)}銘柄)...")
    univ_eval = download_universe(
        all_eval_needed,
        start=EVAL_START, end=EVAL_END,
        verbose=False,
    )
    print(f"  取得完了: {len(univ_eval)} 銘柄")

    # Eval期間のRSR（ポートフォリオ運用中に使うRSR）
    prices_eval = {sym: info["df"]["Close"] for sym, info in univ_eval.items()}
    rsr_eval    = calc_universe_rsr(prices_eval)
    print(f"  RSR計算完了（eval用）")

    # ----------------------------------------------------------------
    # Step 7: 3シナリオ ポートフォリオバックテスト
    # ----------------------------------------------------------------
    print("\n[7/7] 3シナリオ ポートフォリオバックテスト...")
    results = []

    # ---------- BIASED ----------
    univ_b = {sym: univ_eval[sym] for sym in biased_tickers if sym in univ_eval}
    print(f"\n  [BIASED]    {len(univ_b)}銘柄（2018-2024選定・in-sample）...")
    strat_b = {
        sym: make_strategy(sym, univ_b[sym]["sector"], rsr_eval)
        for sym in univ_b
    }
    res_b = run_portfolio(univ_b, strat_b)
    results.append(metrics_row(res_b, f"BIASED({len(univ_b)}銘柄)"))
    m = results[-1]
    print(f"    Sharpe={m['Sharpe']}  CAGR={m['CAGR%']}%  "
          f"MaxDD={m['MaxDD%']}%  Calmar={m['Calmar']}")

    # ---------- TEMPORAL ----------
    univ_t = {sym: univ_eval[sym] for sym in temporal_syms if sym in univ_eval}
    print(f"\n  [TEMPORAL]  {len(univ_t)}銘柄（2015-2017選定・時間的分離）...")
    strat_t = {
        sym: make_strategy(sym, univ_t[sym]["sector"], rsr_eval)
        for sym in univ_t
    }
    res_t = run_portfolio(univ_t, strat_t)
    results.append(metrics_row(res_t, f"TEMPORAL({len(univ_t)}銘柄)"))
    m = results[-1]
    print(f"    Sharpe={m['Sharpe']}  CAGR={m['CAGR%']}%  "
          f"MaxDD={m['MaxDD%']}%  Calmar={m['Calmar']}")

    # ---------- TOP2/SECTOR ----------
    univ_s = {sym: univ_eval[sym] for sym in top2_syms if sym in univ_eval}
    print(f"\n  [TOP2_SEC]  {len(univ_s)}銘柄（出来高上位2/セクター）...")
    strat_s = {
        sym: make_strategy(sym, univ_s[sym]["sector"], rsr_eval)
        for sym in univ_s
    }
    res_s = run_portfolio(univ_s, strat_s)
    results.append(metrics_row(res_s, f"TOP2_SEC({len(univ_s)}銘柄)"))
    m = results[-1]
    print(f"    Sharpe={m['Sharpe']}  CAGR={m['CAGR%']}%  "
          f"MaxDD={m['MaxDD%']}%  Calmar={m['Calmar']}")

    # ================================================================
    # 結果出力
    # ================================================================
    print("\n" + "=" * 72)
    print("  【バイアス定量化結果】")
    print("=" * 72)
    cols = ["シナリオ", "CAGR%", "MaxDD%", "Calmar", "Sharpe",
            "取引数", "勝率%", "平均保有銘柄", "最終資産(万)"]
    df_cmp = pd.DataFrame([{c: m[c] for c in cols} for m in results])
    print(df_cmp.to_string(index=False))

    sh_b = float(results[0]["Sharpe"])
    sh_t = float(results[1]["Sharpe"])
    sh_s = float(results[2]["Sharpe"])

    print("\n  --- バイアス量の算出 ---")
    print(f"  銘柄選択バイアス (BIASED - TEMPORAL) : {sh_b - sh_t:+.3f} Sharpe")
    print(f"  銘柄選択バイアス (BIASED - TOP2_SEC)  : {sh_b - sh_s:+.3f} Sharpe")
    print(f"  セクター効果推定 (TOP2_SEC ≈ セクター選択のみ): Sharpe={sh_s:.3f}")
    print(f"  真の性能下限推定 (TEMPORAL)           : Sharpe={sh_t:.3f}")

    print("\n  --- Phase 1基準（Sharpe>0.5 / MaxDD<20%）との照合 ---")
    for m, label in zip(results, ["BIASED", "TEMPORAL", "TOP2_SEC"]):
        sh  = float(m["Sharpe"])
        mdd = abs(float(m["MaxDD%"]))
        ok  = sh > 0.5 and mdd < 20.0
        mark = "✅" if ok else "❌"
        print(f"  {mark} {label:<12}: Sharpe={sh:.3f}  MaxDD={mdd:.1f}%")

    if sh_t > 0.5:
        print(f"\n  ✅ TEMPORAL Sharpe={sh_t:.3f} > 0.5 → 戦略に実在するアルファあり")
    elif sh_t > 0.3:
        print(f"\n  ⚠  TEMPORAL Sharpe={sh_t:.3f} ∈ (0.3, 0.5) → 弱いが有効な戦略")
    else:
        print(f"\n  ❌ TEMPORAL Sharpe={sh_t:.3f} < 0.3 → 戦略の根本再設計が必要")

    # 年次内訳
    for res_obj, label in [
        (res_b, "BIASED"),
        (res_t, "TEMPORAL"),
        (res_s, "TOP2_SEC"),
    ]:
        print(f"\n  === {label} 年次内訳 ===")
        df_yr = yearly_table(res_obj)
        print(df_yr.to_string(index=False))

    # 選定銘柄の重複確認
    biased_set   = set(biased_tickers.keys())
    temporal_set = set(temporal_syms)
    top2_set     = set(top2_syms)
    common_bt    = biased_set & temporal_set
    common_bs    = biased_set & top2_set
    print(f"\n  --- ユニバース重複 ---")
    print(f"  BIASED∩TEMPORAL : {len(common_bt)}銘柄  {sorted(common_bt)}")
    print(f"  BIASED∩TOP2_SEC : {len(common_bs)}銘柄  {sorted(common_bs)}")
    print(f"  TEMPORAL only    : {sorted(temporal_set - biased_set)}")
    print(f"  BIASED only      : {sorted(biased_set - temporal_set)}")

    # ================================================================
    # チャート出力
    # ================================================================
    colors = {
        results[0]["シナリオ"]: "gray",
        results[1]["シナリオ"]: "steelblue",
        results[2]["シナリオ"]: "darkorange",
    }
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        "時間的分離バックテスト — 銘柄選択バイアスの定量化\n"
        "gray=BIASED(2018-2024選定) / blue=TEMPORAL(2015-2017選定) / orange=TOP2_SEC(出来高)",
        fontsize=10,
    )

    ax = axes[0]
    for m in results:
        r   = m["_res"]
        lbl = m["シナリオ"]
        ax.plot(
            r.equity_curve.index,
            r.equity_curve.values / 10_000,
            color=colors.get(lbl, "purple"),
            linewidth=2.0,
            label=f"{lbl}  Sharpe={m['Sharpe']}  CAGR={m['CAGR%']}%",
        )
    ax.axhline(CAPITAL / 10_000, color="black", linestyle="--", linewidth=0.7,
               label="初期資本")
    ax.set_ylabel("資産（万円）")
    ax.set_title("資産推移比較")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"¥{x:.0f}万"))

    ax = axes[1]
    for m in results:
        r   = m["_res"]
        lbl = m["シナリオ"]
        ax.plot(
            r.dd_series.index,
            r.dd_series.values * 100,
            color=colors.get(lbl, "purple"),
            linewidth=1.8,
            label=f"{lbl} MaxDD={m['MaxDD%']}%",
        )
    ax.axhline(-15, color="darkred", linestyle="--", linewidth=1.0, label="CB -15%")
    ax.set_ylabel("ドローダウン（%）")
    ax.set_title("ドローダウン比較")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    out_png = os.path.join(OUTPUT, "temporal_separation_bias.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  チャート保存: {out_png}")

    # ================================================================
    # CSV保存
    # ================================================================
    df_cmp.to_csv(
        os.path.join(OUTPUT, "temporal_separation_bias.csv"),
        index=False, encoding="utf-8-sig",
    )

    # 選定銘柄リスト
    rows_sym = []
    for sym in sorted(biased_tickers):
        rows_sym.append({
            "シナリオ": "BIASED",
            "銘柄":     sym,
            "セクター": biased_tickers[sym],
            "TEMPORAL": "✅" if sym in temporal_set else "",
            "TOP2_SEC": "✅" if sym in top2_set else "",
        })
    for sym in sorted(temporal_set - biased_set):
        rows_sym.append({
            "シナリオ": "TEMPORAL_ONLY",
            "銘柄":     sym,
            "セクター": TOPIX100_TICKERS.get(sym, ""),
            "TEMPORAL": "✅",
            "TOP2_SEC": "✅" if sym in top2_set else "",
        })
    df_syms = pd.DataFrame(rows_sym)
    df_syms.to_csv(
        os.path.join(OUTPUT, "temporal_separation_universes.csv"),
        index=False, encoding="utf-8-sig",
    )

    print("\n完了。")
    return results


if __name__ == "__main__":
    main()
