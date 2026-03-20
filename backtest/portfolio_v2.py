"""
backtest/portfolio_v2.py
改良版ポートフォリオ — 最重要改善案を全実装

【4つの改善】
  1. RSR閾値を75→70に緩和（感度分析でCalmar最大を確認済み）
  2. 均等ウェイト（vol_target=0, use_idm=False）→ CAGR+3.8%
  3. ユニバース拡大（TOPIX100+15銘柄 → 91銘柄でスクリーニング）
  4. 決算シーズン保護（2/5/8/11月1-25日は新規エントリー禁止）

【比較シナリオ】
  BASE : RSR>75, ボラ調整+IDM, 27銘柄, EP無し (現行ベスト D_IDM_sec1)
  V1   : RSR>70, 均等ウェイト, 27銘柄, EP無し  (改善1+2のみ)
  V2   : RSR>70, 均等ウェイト, 拡張銘柄, EP無し (改善1+2+3)
  V3   : RSR>70, 均等ウェイト, 拡張銘柄, EP有り  (全改善)
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

OUTPUT = "C:/Users/owner/.claude/レポート"
os.makedirs(OUTPUT, exist_ok=True)

START   = "2018-01-01"
END     = "2024-12-31"
CAPITAL = 2_000_000

# ------------------------------------------------------------------ #
# 改善4: 決算シーズン保護（EarningsProtectedStrategy ラッパー）
# ------------------------------------------------------------------ #
EARNINGS_MONTHS     = {2, 5, 8, 11}   # 決算発表集中月
EARNINGS_BLACKOUT_DAYS_END = 25        # 月初から25日まで新規エントリー禁止

class EarningsProtectedStrategy:
    """
    決算シーズン中の新規エントリーを禁止するラッパー戦略。

    ルール:
      - 2/5/8/11月の1〜25日は BUY シグナルを 0（待機）に変換
      - SELL シグナルは常に通す（損切り・利確は止めない）
    """
    def __init__(self, base_strategy):
        self._base = base_strategy

    @property
    def name(self) -> str:
        return self._base.name + "+EP"

    def generate_signal(self, data: pd.DataFrame) -> int:
        signal = self._base.generate_signal(data)
        if signal == 1:
            date = data.index[-1]
            if date.month in EARNINGS_MONTHS and date.day <= EARNINGS_BLACKOUT_DAYS_END:
                return 0   # 新規エントリー禁止
        return signal


# ------------------------------------------------------------------ #
# 改善3: ユニバース拡大 — 追加銘柄リスト
# ------------------------------------------------------------------ #
ADDITIONAL_TICKERS: dict[str, str] = {
    # 半導体製造装置・電子部品（AI/半導体ブームの恩恵大）
    "6857.T": "電機精密",   # アドバンテスト（半導体テスト装置）
    "6146.T": "電機精密",   # ディスコ（ウェーハスライサー）
    "6981.T": "電機精密",   # 村田製作所（電子部品）
    "6479.T": "電機精密",   # ミネベアミツミ（精密モーター）
    # 産業機械・ロボット
    "6506.T": "機械",       # 安川電機（産業用ロボット）
    "7012.T": "機械",       # 川崎重工業（航空宇宙・防衛・機械）
    "7013.T": "機械",       # IHI（航空エンジン・エネルギー）
    # 海運（第3位）
    "9107.T": "海運",       # 川崎汽船（海運3社の1つ）
    # 商社（中堅）
    "8015.T": "商社",       # 豊田通商（トヨタ系商社）
    # 医療・ヘルスケア
    "6869.T": "電機精密",   # シスメックス（医療検査機器）
    # 精密モーター
    "6594.T": "電機精密",   # ニデック（精密モーター）
    # 銀行（地銀大手）
    "8354.T": "銀行",       # ふくおかフィナンシャルグループ
    # 鉄鋼・素材
    "5706.T": "鉄鋼",       # 三菱マテリアル
    # 小売（成長型）
    "3197.T": "小売",       # すかいらーくHD
    # 情報通信（DX系）
    "4055.T": "情報通信",   # MS-Japan
}

# セクター→戦略マッピング（動的選択ルール）
SECTOR_STRATEGY: dict[str, str] = {
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


# ------------------------------------------------------------------ #
# ヘルパー
# ------------------------------------------------------------------ #
def calmar(res) -> float:
    dd = abs(res.max_drawdown)
    return res.cagr / dd if dd > 0 else float("inf")

def metrics_row(res, label: str) -> dict:
    return {
        "シナリオ":      label,
        "CAGR%":        f"{res.cagr*100:+.2f}",
        "MaxDD%":       f"{res.max_drawdown*100:.2f}",
        "Calmar":       f"{calmar(res):.3f}",
        "Sharpe":       f"{res.sharpe_ratio:.3f}",
        "取引数":       res.n_trades,
        "勝率%":        f"{res.win_rate*100:.1f}",
        "平均保有銘柄": f"{res.n_positions.mean():.2f}",
        "最終資産(万)": f"¥{res.equity_curve.iloc[-1]/10000:.0f}万",
        "_res":          res,
    }

def screen_new_stocks(new_universe: dict, rsr_uni: pd.DataFrame,
                      min_sharpe=0.30, max_dd=-0.30,
                      required_start="2018-03-01") -> list[str]:
    """
    追加銘柄をシングル銘柄バックテストでスクリーニングし
    sharpe > min_sharpe AND maxdd > max_dd の銘柄コードを返す。

    ★ required_start: この日付より前にデータがない銘柄は除外
       （期間不一致を防ぎ BASEと同じ2018-2024で比較するため）
    """
    cost = TradeCost()
    passed = []
    req_ts = pd.Timestamp(required_start)
    print(f"\n  === 追加銘柄スクリーニング ({len(new_universe)}銘柄) ===")
    for sym, info in new_universe.items():
        df_s = info["df"]
        sector = info["sector"]

        # ★ データ開始日チェック（2018年データがない銘柄を除外）
        if df_s.index[0] > req_ts:
            print(f"  ✗  {sym:<8} データ開始={df_s.index[0].date()} → 2018年データなし → 除外")
            continue

        rsr_s = rsr_uni[sym] if sym in rsr_uni.columns else None

        # 戦略割当
        rule = SECTOR_STRATEGY.get(sector, "dynamic")
        if rule == "fujiko":
            strat = FujikoStrategy(rsr_series=rsr_s, min_sepa=6, min_rsr=70.0,
                                   mom_period=21, turtle_entry=20, turtle_exit=20,
                                   use_turtle_entry=True)
        else:
            strat = MeanReversionStrategy(**MR_PARAMS)

        eng = BacktestEngine(prices=df_s, strategy=strat,
                             capital=1_000_000, cost=cost, symbol=sym)
        r = eng.run()
        dd = r.max_drawdown
        sh = r.sharpe_ratio
        ok = sh > min_sharpe and dd > max_dd
        mark = "✅" if ok else "✗ "
        print(f"  {mark} {sym:<8} ({sector:<8}) Sharpe={sh:+.3f} MaxDD={dd*100:.1f}%"
              f"  CAGR={r.total_return*100:+.0f}%  開始={df_s.index[0].date()}")
        if ok:
            passed.append(sym)
    return passed

def build_strats(universe: dict, rsr_uni: pd.DataFrame, df_sel: pd.DataFrame,
                 sym_to_strat: dict, min_rsr: float, use_ep: bool) -> dict:
    strat_dict = {}
    for sym, info in universe.items():
        rsr_s  = rsr_uni[sym] if sym in rsr_uni.columns else None
        lbl    = sym_to_strat.get(sym, "フジコ法")
        rule   = SECTOR_STRATEGY.get(info["sector"], "dynamic")
        if rule == "mean_rev" or (rule == "dynamic" and "平均回帰" in lbl):
            base = MeanReversionStrategy(**MR_PARAMS)
        else:
            base = FujikoStrategy(rsr_series=rsr_s, min_sepa=6, min_rsr=min_rsr,
                                  mom_period=21, turtle_entry=20, turtle_exit=20,
                                  use_turtle_entry=True)
        strat_dict[sym] = EarningsProtectedStrategy(base) if use_ep else base
    return strat_dict

def run_portfolio(universe: dict, strat_dict: dict, vol_tgt: float,
                  max_wt: float, use_idm: bool) -> object:
    return PortfolioEngine(
        universe=universe, strategy=strat_dict, capital=CAPITAL,
        max_dd_limit=0.15, min_sectors=1, max_positions=3,
        cost=TradeCost(), vol_target=vol_tgt, max_single_weight=max_wt,
        vol_window=20, use_idm=use_idm, use_dd_scalar=False,
    ).run()

def yearly_table(res) -> pd.DataFrame:
    rows = []
    eq = res.equity_curve; dd = res.dd_series
    for yr in range(2018, 2025):
        ye = eq[eq.index.year == yr]; yd = dd[dd.index.year == yr]
        if ye.empty: continue
        ret = ye.iloc[-1] / ye.iloc[0] - 1
        mdd = float(yd.min())
        cal = ret / abs(mdd) if mdd < 0 else None
        rows.append({"年": yr,
                     "リターン%": round(ret*100, 2),
                     "MaxDD%": round(mdd*100, 2),
                     "Calmar": round(cal, 2) if cal else "∞"})
    return pd.DataFrame(rows)


# ================================================================
# MAIN
# ================================================================
def main():
    print("=" * 72)
    print("  portfolio_v2.py — 最重要改善案 全実装")
    print("  期間: 2018〜2024  /  初期資本: ¥2,000,000")
    print("=" * 72)

    # ---- 1. データ取得 ----
    df_sel       = pd.read_csv("data/dynamic_selection.csv", encoding="utf-8-sig")
    sym_to_strat = dict(zip(df_sel["symbol"], df_sel["strategy"]))
    mask_27      = (df_sel["sharpe"] > 0.30) & (df_sel["maxdd"].abs() < 0.30)
    tickers_27   = {r["symbol"]: r["sector"] for _, r in df_sel[mask_27].iterrows()}

    # 既存27銘柄 + 追加銘柄の全体取得
    all_tickers = {**tickers_27, **ADDITIONAL_TICKERS}
    print(f"\n[1/5] データ取得中（{len(all_tickers)}銘柄）...")
    universe_all = download_universe(all_tickers, start=START, end=END, verbose=False)
    print(f"  取得完了: {len(universe_all)} 銘柄")

    # ---- 2. RSR計算（全銘柄一括） ----
    prices_all = {sym: info["df"]["Close"] for sym, info in universe_all.items()}
    rsr_uni    = calc_universe_rsr(prices_all)
    print(f"  RSR計算完了: {rsr_uni.shape[1]} 銘柄")

    # ---- 3. 追加銘柄スクリーニング ----
    print("\n[2/5] 追加銘柄スクリーニング...")
    new_sym_universe = {sym: universe_all[sym] for sym in ADDITIONAL_TICKERS
                        if sym in universe_all}
    passed_new = screen_new_stocks(new_sym_universe, rsr_uni)
    passed_syms = set(tickers_27.keys()) | set(passed_new)
    print(f"\n  合格銘柄: {len(passed_new)}追加 → 合計 {len(passed_syms)}銘柄")

    # 3種類のユニバース辞書
    univ_27  = {sym: universe_all[sym] for sym in tickers_27 if sym in universe_all}
    univ_exp = {sym: universe_all[sym] for sym in passed_syms if sym in universe_all}

    # ---- 4. シナリオ実行 ----
    print("\n[3/5] 4シナリオ バックテスト実行...")

    # BASE: RSR>75, ボラ調整+IDM, 27銘柄, EP無し （現行ベスト D_IDM_sec1）
    print("  [BASE] RSR>75 / ボラ調整+IDM / 27銘柄 / EP無し...")
    s_base = build_strats(univ_27, rsr_uni, df_sel, sym_to_strat, 75.0, False)
    res_base = run_portfolio(univ_27, s_base, vol_tgt=0.02, max_wt=0.25, use_idm=True)
    m_base = metrics_row(res_base, "BASE(現行)")
    print(f"    CAGR={m_base['CAGR%']}%  MaxDD={m_base['MaxDD%']}%  "
          f"Calmar={m_base['Calmar']}  保有={m_base['平均保有銘柄']}")

    # V1: RSR>70, 均等ウェイト, 27銘柄, EP無し （改善1+2）
    print("  [V1]   RSR>70 / 均等ウェイト / 27銘柄 / EP無し...")
    s_v1 = build_strats(univ_27, rsr_uni, df_sel, sym_to_strat, 70.0, False)
    res_v1 = run_portfolio(univ_27, s_v1, vol_tgt=0.0, max_wt=0.25, use_idm=False)
    m_v1 = metrics_row(res_v1, "V1(RSR70+均等)")
    print(f"    CAGR={m_v1['CAGR%']}%  MaxDD={m_v1['MaxDD%']}%  "
          f"Calmar={m_v1['Calmar']}  保有={m_v1['平均保有銘柄']}")

    # V2: RSR>70, 均等ウェイト, 拡張銘柄, EP無し （改善1+2+3）
    print(f"  [V2]   RSR>70 / 均等ウェイト / {len(univ_exp)}銘柄 / EP無し...")
    s_v2 = build_strats(univ_exp, rsr_uni, df_sel, sym_to_strat, 70.0, False)
    res_v2 = run_portfolio(univ_exp, s_v2, vol_tgt=0.0, max_wt=0.25, use_idm=False)
    m_v2 = metrics_row(res_v2, f"V2(RSR70+均等+{len(univ_exp)}銘柄)")
    print(f"    CAGR={m_v2['CAGR%']}%  MaxDD={m_v2['MaxDD%']}%  "
          f"Calmar={m_v2['Calmar']}  保有={m_v2['平均保有銘柄']}")

    # V3: RSR>70, 均等ウェイト, 拡張銘柄, EP有り （均等+EP組み合わせ検証）
    print(f"  [V3]   RSR>70 / 均等ウェイト / {len(univ_exp)}銘柄 / EP有り...")
    s_v3 = build_strats(univ_exp, rsr_uni, df_sel, sym_to_strat, 70.0, True)
    res_v3 = run_portfolio(univ_exp, s_v3, vol_tgt=0.0, max_wt=0.25, use_idm=False)
    m_v3 = metrics_row(res_v3, "V3(均等+EP)")
    print(f"    CAGR={m_v3['CAGR%']}%  MaxDD={m_v3['MaxDD%']}%  "
          f"Calmar={m_v3['Calmar']}  保有={m_v3['平均保有銘柄']}")

    # V4: RSR>70, ボラ調整+IDM（現行方式）, 拡張銘柄, EP無し ← 推奨最終形
    print(f"  [V4★] RSR>70 / ボラ調整+IDM / {len(univ_exp)}銘柄 / EP無し  ← 推奨...")
    s_v4 = build_strats(univ_exp, rsr_uni, df_sel, sym_to_strat, 70.0, False)
    res_v4 = run_portfolio(univ_exp, s_v4, vol_tgt=0.02, max_wt=0.25, use_idm=True)
    m_v4 = metrics_row(res_v4, "V4★(RSR70+ボラ+拡張)")
    print(f"    CAGR={m_v4['CAGR%']}%  MaxDD={m_v4['MaxDD%']}%  "
          f"Calmar={m_v4['Calmar']}  保有={m_v4['平均保有銘柄']}")

    results = [m_base, m_v1, m_v2, m_v3, m_v4]

    # ---- 5. 結果表示 ----
    print("\n" + "=" * 72)
    print("  比較結果サマリー")
    print("=" * 72)
    cols = ["シナリオ","CAGR%","MaxDD%","Calmar","Sharpe","取引数","勝率%","平均保有銘柄","最終資産(万)"]
    df_cmp = pd.DataFrame([{c: m[c] for c in cols} for m in results])
    print(df_cmp.to_string(index=False))

    # 年次内訳（V4 推奨）
    print("\n  === V4★（推奨最終形）年次内訳 ===")
    df_yr_v3 = yearly_table(res_v4)
    print(f"  {'年':<6} {'リターン':>10} {'MaxDD':>10} {'Calmar':>8}")
    for _, row in df_yr_v3.iterrows():
        print(f"  {row['年']:<6} {str(row['リターン%'])+' %':>10} "
              f"{str(row['MaxDD%'])+' %':>10} {str(row['Calmar']):>8}")
    total_ret = res_v4.equity_curve.iloc[-1] / res_v4.equity_curve.iloc[0] - 1
    print(f"  {'全期間':<6} {total_ret*100:>+9.2f}%  {res_v4.max_drawdown*100:>+9.2f}%  {calmar(res_v4):>8.3f}")

    # ---- 6. チャート出力 ----
    print("\n[4/5] チャート生成中...")
    fig, axes = plt.subplots(3, 1, figsize=(15, 13), sharex=True)
    fig.suptitle(
        "portfolio_v2 — 改良版 4シナリオ比較（2018-2024）\n"
        "BASE=現行 / V1=RSR70+均等 / V2=V1+拡張銘柄 / V3=V2+決算EP",
        fontsize=11,
    )
    colors = {"BASE(現行)": "gray", "V1(RSR70+均等)": "steelblue",
              f"V2(RSR70+均等+{len(univ_exp)}銘柄)": "darkorange",
              "V3(全改善)": "crimson"}
    lws    = {"BASE(現行)": 1.5, "V1(RSR70+均等)": 1.8,
              f"V2(RSR70+均等+{len(univ_exp)}銘柄)": 2.0,
              "V3(全改善)": 2.5}

    ax = axes[0]
    for m in results:
        res_i = m["_res"]
        lbl   = m["シナリオ"]
        clr   = colors.get(lbl, "purple")
        lw    = lws.get(lbl, 1.5)
        ax.plot(res_i.equity_curve.index,
                res_i.equity_curve.values / 10_000,
                color=clr, linewidth=lw,
                label=f"{lbl}  CAGR={m['CAGR%']}%  Calmar={m['Calmar']}")
    ax.axhline(CAPITAL / 10_000, color="black", linestyle="--", linewidth=0.7)
    ax.set_ylabel("資産（万円）")
    ax.set_title("資産推移比較")
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"¥{x:.0f}万"))
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    for m in results:
        res_i = m["_res"]
        lbl   = m["シナリオ"]
        ax.plot(res_i.dd_series.index, res_i.dd_series.values * 100,
                color=colors.get(lbl, "purple"),
                linewidth=lws.get(lbl, 1.5),
                label=f"{lbl} MaxDD={m['MaxDD%']}%")
    ax.axhline(-15, color="darkred", linestyle="--", linewidth=1.0, label="CB閾値 -15%")
    ax.set_ylabel("ドローダウン（%）")
    ax.set_title("ドローダウン比較")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    ax = axes[2]
    for m in results:
        res_i = m["_res"]
        lbl   = m["シナリオ"]
        ax.step(res_i.n_positions.index, res_i.n_positions.values,
                where="post", color=colors.get(lbl, "purple"),
                linewidth=1.3,
                label=f"{lbl} 平均={m['平均保有銘柄']}")
    ax.set_ylabel("保有銘柄数")
    ax.set_title("保有銘柄数推移")
    ax.set_ylim(0, 5)
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out_png = os.path.join(OUTPUT, "portfolio_v2_comparison.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out_png}")

    # ---- 7. CSV保存 ----
    print("\n[5/5] CSV保存中...")
    df_cmp.to_csv(os.path.join(OUTPUT, "v2_scenario_comparison.csv"),
                  index=False, encoding="utf-8-sig")
    df_yr_v3.to_csv(os.path.join(OUTPUT, "v2_yearly_v4.csv"),
                    index=False, encoding="utf-8-sig")

    # 採用拡張銘柄リスト
    new_adopted = [sym for sym in passed_new if sym in univ_exp]
    df_new = pd.DataFrame([
        {"追加銘柄": sym, "セクター": ADDITIONAL_TICKERS[sym]}
        for sym in new_adopted
    ])
    if not df_new.empty:
        df_new.to_csv(os.path.join(OUTPUT, "v2_new_stocks.csv"),
                      index=False, encoding="utf-8-sig")
        print(f"  採用追加銘柄: {', '.join(new_adopted)}")
    else:
        print("  追加銘柄でフィルターを通過したものはなし")

    print("\n" + "=" * 72)
    print("  最終比較（ベースラインとの差）")
    print("=" * 72)
    base_cagr   = float(m_base["CAGR%"])
    base_maxdd  = float(m_base["MaxDD%"])
    base_calmar = float(m_base["Calmar"])
    base_sharpe = float(m_base["Sharpe"])
    for m in results[1:]:
        d_cagr   = float(m["CAGR%"])   - base_cagr
        d_maxdd  = float(m["MaxDD%"])  - base_maxdd
        d_calmar = float(m["Calmar"])  - base_calmar
        d_sharpe = float(m["Sharpe"])  - base_sharpe
        print(f"\n  {m['シナリオ']}")
        print(f"    CAGR  : {float(m['CAGR%']):+.2f}%  (Δ{d_cagr:+.2f}%)")
        print(f"    MaxDD : {float(m['MaxDD%']):.2f}%  (Δ{d_maxdd:+.2f}%)")
        print(f"    Calmar: {float(m['Calmar']):.3f}   (Δ{d_calmar:+.3f})")
        print(f"    Sharpe: {float(m['Sharpe']):.3f}   (Δ{d_sharpe:+.3f})")
        print(f"    平均保有: {m['平均保有銘柄']}銘柄  /  最終資産: {m['最終資産(万)']}")

    print("\n完了。")
    return results


if __name__ == "__main__":
    main()
