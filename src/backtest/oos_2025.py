"""
backtest/oos_2025.py
2025年 OOS（Out-of-Sample）検証

【設計方針】
  - ユニバース: configs/universe/2026Q1_temporal24.json（固定・再選定なし）
    = 2015-2017 Selection期間で選定した24銘柄
  - IS期間:    2018-01-01 〜 2024-12-31（比較用）
  - OOS期間:   2025-01-01 〜 2025-12-31（真のOOS）
  - パラメータ: portfolio_temporal_separation.py と完全一致（変更禁止）

【診断指標（TEMPORAL特有の3点）】
  1. avg_exposure         : 実稼働率（理想: 0.12–0.28）
  2. trades_per_month     : 月次エントリー頻度（正常: 2–6 trades/month）
  3. days_to_recover_peak : DD回復速度（> 140日はボトルネック）
"""

from __future__ import annotations
import os, sys, json, warnings
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

from backtest.universe_builder        import download_universe
from backtest.rsr                     import calc_universe_rsr
from backtest.fujiko_strategy         import FujikoStrategy
from backtest.mean_reversion_strategy import MeanReversionStrategy
from backtest.portfolio_engine        import PortfolioEngine
from backtest.engine                  import BacktestEngine, TradeCost

# ------------------------------------------------------------------ #
# 定数（変更禁止: IS比較と完全に揃えるため）
# ------------------------------------------------------------------ #
UNIVERSE_JSON = "configs/universe/2026Q1_temporal24.json"
IS_START   = "2018-01-01"
IS_END     = "2024-12-31"
OOS_START  = "2025-01-01"
OOS_END    = "2025-12-31"
CAPITAL    = 2_000_000
OUTPUT     = "C:/ai-trading/reports"
os.makedirs(OUTPUT, exist_ok=True)

# セクター→戦略マッピング（portfolio_temporal_separation.py と同一）
SECTOR_STRATEGY = {
    "化学":    "fujiko",  "医薬品":  "fujiko",
    "レジャー":"fujiko",  "石油":    "fujiko",
    "鉄鋼":    "mean_rev","電機":    "fujiko",
    "電機精密":"fujiko",  "商社":    "fujiko",
    "小売":    "mean_rev","銀行":    "mean_rev",
    "サービス":"fujiko",  "陸運":    "fujiko",
}
MR_PARAMS = dict(
    rsi_period=5, rsi_entry=25.0, rsi_exit=65.0,
    ma_long=200, stop_loss_pct=0.07, max_hold_days=10,
)


# ------------------------------------------------------------------ #
# ヘルパー
# ------------------------------------------------------------------ #
def make_strategy(sym: str, sector: str, rsr_uni: pd.DataFrame):
    rsr_s = rsr_uni[sym] if sym in rsr_uni.columns else None
    rule  = SECTOR_STRATEGY.get(sector, "fujiko")
    if rule == "mean_rev":
        return MeanReversionStrategy(**MR_PARAMS)
    return FujikoStrategy(
        rsr_series=rsr_s, min_sepa=6, min_rsr=70.0,
        mom_period=21, turtle_entry=20, turtle_exit=20,
        use_turtle_entry=True,
    )


def run_portfolio(universe_eval: dict, strat_dict: dict):
    return PortfolioEngine(
        universe=universe_eval,
        strategy=strat_dict,
        capital=CAPITAL,
        max_dd_limit=0.15,
        min_sectors=1,
        max_positions=3,
        cost=TradeCost(),
        vol_target=0.0,
        max_single_weight=0.25,
        vol_window=20,
        use_idm=False,
        use_dd_scalar=False,
    ).run()


def calmar(res) -> float:
    mdd = abs(res.max_drawdown)
    return res.cagr / mdd if mdd > 0 else float("inf")


def compute_exposure(res) -> float:
    """平均稼働率: ポジション保有日数 / 総日数"""
    pos = res.n_positions  # pandas Series
    return float((pos > 0).mean())


def compute_trades_per_month(res, period_start: str, period_end: str) -> float:
    """月次エントリー頻度（取引数 / 期間月数）"""
    months = (pd.Timestamp(period_end) - pd.Timestamp(period_start)).days / 30.44
    return res.n_trades / months if months > 0 else 0.0


def compute_days_to_recover(res) -> float:
    """DDの平均回復日数（DD < -1% 区間の継続日数の平均）"""
    dd = res.dd_series
    in_dd = (dd < -0.01)
    if not in_dd.any():
        return 0.0
    # 連続区間を特定
    durations = []
    count = 0
    for v in in_dd:
        if v:
            count += 1
        else:
            if count > 0:
                durations.append(count)
            count = 0
    if count > 0:
        durations.append(count)
    return float(np.mean(durations)) if durations else 0.0


def yearly_table(res, start_yr: int, end_yr: int) -> pd.DataFrame:
    rows = []
    eq = res.equity_curve
    dd = res.dd_series
    for yr in range(start_yr, end_yr + 1):
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


def print_diagnosis(label: str, res, period_start: str, period_end: str):
    """TEMPORAL特有の診断3点を出力"""
    avg_exp  = compute_exposure(res)
    tpm      = compute_trades_per_month(res, period_start, period_end)
    d2r      = compute_days_to_recover(res)

    exp_ok  = "✅" if 0.12 <= avg_exp <= 0.28 else ("⚠ 低" if avg_exp < 0.12 else "⚠ 高")
    tpm_ok  = "✅" if 2 <= tpm <= 6 else ("⚠ 少" if tpm < 2 else "⚠ 多")
    d2r_ok  = "✅" if d2r <= 140 else "⚠ ボトルネック"

    print(f"\n  --- {label} 診断 ---")
    print(f"  avg_exposure      : {avg_exp:.3f}  {exp_ok}  (理想: 0.12–0.28)")
    print(f"  trades/month      : {tpm:.2f}   {tpm_ok}  (正常: 2–6)")
    print(f"  days_to_recover   : {d2r:.1f}日 {d2r_ok}  (< 140日)")


def pattern_diagnosis(sharpe: float, avg_exp: float, label: str):
    """パターンA/B/C診断"""
    print(f"\n  --- {label} パターン診断 ---")
    if sharpe >= 0.8 and sharpe <= 1.2:
        print(f"  ✅ パターンA（理想）: Sharpe={sharpe:.3f} ∈ [0.8, 1.2]")
        print(f"     → 戦略は構造的に安定")
    elif avg_exp < 0.10:
        print(f"  ❌ パターンB（危険）: avg_exposure={avg_exp:.3f} < 0.10")
        print(f"     → フィルタが強すぎる / rare-event capture に近い")
    elif sharpe > 1.6:
        print(f"  🔔 パターンC（要注意）: Sharpe={sharpe:.3f} > 1.6")
        print(f"     → IS期間(2018-2024)が逆に不利だった可能性")
        print(f"     → または2025年が特異的に良好な相場環境")
    elif sharpe < 0.3:
        print(f"  ❌ パターン外: Sharpe={sharpe:.3f} < 0.3")
        print(f"     → 戦略崩壊の可能性。根本再設計を検討")
    else:
        print(f"  ⚠ 中間域: Sharpe={sharpe:.3f} ∈ (0.3, 0.8)")
        print(f"     → 弱いが有効。セクター構成か市場レジームを確認")


# ================================================================
# MAIN
# ================================================================
def main():
    print("=" * 72)
    print("  backtest/oos_2025.py — 2025年 OOS検証")
    print(f"  Universe  : {UNIVERSE_JSON}（固定・再選定なし）")
    print(f"  IS期間    : {IS_START} 〜 {IS_END}")
    print(f"  OOS期間   : {OOS_START} 〜 {OOS_END}")
    print("=" * 72)

    cost = TradeCost()

    # ----------------------------------------------------------------
    # Step 1: ユニバース読み込み（固定）
    # ----------------------------------------------------------------
    print(f"\n[1/5] ユニバース読み込み ({UNIVERSE_JSON})...")
    with open(UNIVERSE_JSON, encoding="utf-8") as f:
        uni_cfg = json.load(f)
    fixed_tickers: dict[str, str] = uni_cfg["symbols"]
    print(f"  銘柄数: {len(fixed_tickers)}")
    for sym, sec in fixed_tickers.items():
        print(f"    {sym:<8} {sec}")

    # ----------------------------------------------------------------
    # Step 2: 2018-2025 通しでデータ取得（RSRウォームアップ確保）
    # ----------------------------------------------------------------
    # ★ 設計根拠:
    #   RSRはIBD式12ヶ月加重リターン（252日ウォームアップが必要）。
    #   OOS期間(2025)だけでRSRを計算すると全期間NaN → avg_exposure≈0 になる。
    #   正しい設計: 2018-2025 通しで実行 → 2025部分の結果のみを抜き出す。
    #   これは「モデルの構造は固定・パラメータは変更なし」であり真のOOSを保つ。
    FULL_END = "2025-12-31"

    # RSRユニバース = TEMPORAL24 固定（research = live = identical）
    # ★ 設計方針: RSR計算・売買対象ともに同一の24銘柄を使う。
    #   これにより研究環境とライブ環境のRSRパーセンタイルが完全一致する。

    print(f"\n[2/5] 通しデータ取得 ({IS_START}〜{FULL_END})...")
    univ_full = download_universe(fixed_tickers, start=IS_START, end=FULL_END, verbose=False)
    print(f"  取得完了: {len(univ_full)} 銘柄（RSR計算 = 売買対象 = TEMPORAL24）")

    # RSR: TEMPORAL24の24銘柄で計算
    prices_full = {sym: info["df"]["Close"] for sym, info in univ_full.items()}
    rsr_full    = calc_universe_rsr(prices_full)
    print(f"  RSR計算完了: {rsr_full.shape[1]} 銘柄ベース（固定）")

    strat_full  = {sym: make_strategy(sym, univ_full[sym]["sector"], rsr_full) for sym in univ_full}

    # ----------------------------------------------------------------
    # Step 3: 2018-2025 通しでポートフォリオ実行 → IS/OOS分割
    # ----------------------------------------------------------------
    print(f"\n[3/5] 通しポートフォリオバックテスト ({IS_START}〜{FULL_END})...")
    res_full = run_portfolio(univ_full, strat_full)

    # IS: equity_curveの2018-2024部分でメトリクスを再計算
    eq_full  = res_full.equity_curve
    dd_full  = res_full.dd_series
    pos_full = res_full.n_positions

    eq_is  = eq_full[eq_full.index < "2025-01-01"]
    eq_oos = eq_full[eq_full.index >= "2025-01-01"]
    dd_is  = dd_full[dd_full.index < "2025-01-01"]
    dd_oos = dd_full[dd_full.index >= "2025-01-01"]
    pos_is  = pos_full[pos_full.index < "2025-01-01"]
    pos_oos = pos_full[pos_full.index >= "2025-01-01"]

    def _sharpe_from_equity(eq: pd.Series, periods_per_year: float = 252) -> float:
        ret = eq.pct_change().dropna()
        if ret.std() == 0:
            return 0.0
        return float(ret.mean() / ret.std() * np.sqrt(periods_per_year))

    def _cagr_from_equity(eq: pd.Series) -> float:
        if len(eq) < 2:
            return 0.0
        years = len(eq) / 252
        return float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1) * 100

    def _maxdd_from_equity(eq: pd.Series) -> float:
        roll_max = eq.expanding().max()
        dd = (eq / roll_max - 1)
        return float(dd.min()) * 100

    is_sharpe = _sharpe_from_equity(eq_is)
    is_cagr   = _cagr_from_equity(eq_is)
    is_mdd    = _maxdd_from_equity(eq_is)
    is_calmar_v = is_cagr / abs(is_mdd) if is_mdd < 0 else float("inf")
    is_exp    = float((pos_is > 0).mean())

    oos_sharpe = _sharpe_from_equity(eq_oos)
    oos_cagr   = _cagr_from_equity(eq_oos)
    oos_mdd    = _maxdd_from_equity(eq_oos)
    oos_calmar = oos_cagr / abs(oos_mdd) if oos_mdd < 0 else float("inf")
    oos_exp    = float((pos_oos > 0).mean())

    # n_trades / win_rate はresオブジェクトから直接取れないため全体で代用
    oos_n_trades = int(res_full.n_trades)  # 近似（2025分を個別カウントは不可）

    # res_is / res_oos オブジェクトを後続の診断のために作る（軽量struct）
    class _Slice:
        def __init__(self, eq, dd, pos):
            self.equity_curve = eq
            self.dd_series    = dd
            self.n_positions  = pos
            self.n_trades     = 0  # 診断では使わない
            self.win_rate     = 0.0

    res_is  = _Slice(eq_is,  dd_is,  pos_is)
    res_oos = _Slice(eq_oos, dd_oos, pos_oos)

    print(f"  IS  → Sharpe={is_sharpe:.3f}  CAGR={is_cagr:+.2f}%  "
          f"MaxDD={is_mdd:.2f}%  Calmar={is_calmar_v:.3f}  avg_exp={is_exp:.3f}")
    print(f"  OOS → Sharpe={oos_sharpe:.3f}  CAGR={oos_cagr:+.2f}%  "
          f"MaxDD={oos_mdd:.2f}%  Calmar={oos_calmar:.3f}  avg_exp={oos_exp:.3f}")

    # ----------------------------------------------------------------
    # Step 4: 結果出力
    # ----------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  【OOS検証結果】")
    print("=" * 72)
    print(f"\n  {'指標':<20} {'IS(2018-2024)':<20} {'OOS(2025)':<20}")
    print(f"  {'-'*18:<20} {'-'*18:<20} {'-'*18:<20}")
    print(f"  {'CAGR':<20} {is_cagr:>+8.2f}%            {oos_cagr:>+8.2f}%")
    print(f"  {'Sharpe':<20} {is_sharpe:>8.3f}             {oos_sharpe:>8.3f}")
    print(f"  {'MaxDD':<20} {is_mdd:>8.2f}%            {oos_mdd:>8.2f}%")
    print(f"  {'Calmar':<20} {is_calmar_v:>8.3f}             {oos_calmar:>3.3f}")
    print(f"  {'avg_exposure':<20} {is_exp:>8.3f}             {oos_exp:>8.3f}")

    # TEMPORAL基準値との照合
    print(f"\n  --- OOS vs TEMPORAL基準（Sharpe≈1.07想定）---")
    if 0.7 <= oos_sharpe <= 1.3:
        print(f"  ✅ Sharpe={oos_sharpe:.3f} ∈ [0.7, 1.3]  → 期待通り")
    elif oos_sharpe < 0.3:
        print(f"  ❌ Sharpe={oos_sharpe:.3f} < 0.3  → 戦略崩壊を疑う")
    elif oos_sharpe < 0.7:
        print(f"  ⚠ Sharpe={oos_sharpe:.3f} ∈ (0.3, 0.7)  → 弱い。市場環境を確認")
    else:
        print(f"  🔔 Sharpe={oos_sharpe:.3f} > 1.3  → 良すぎる。2025が特異か確認")

    # Phase 1基準
    ph1_ok = oos_sharpe > 0.5 and abs(oos_mdd) < 20.0
    print(f"  Phase1基準(Sharpe>0.5, MaxDD<20%): {'✅ クリア' if ph1_ok else '❌ 未達'}")

    # 診断3点
    print_diagnosis("OOS(2025)", res_oos, OOS_START, OOS_END)
    print_diagnosis("IS(2018-2024)", res_is, IS_START, IS_END)

    # パターン診断
    pattern_diagnosis(oos_sharpe, oos_exp, "OOS(2025)")

    # 年次内訳（IS）
    print("\n  === IS 年次内訳（2018-2024）===")
    df_is_yr = yearly_table(res_is, 2018, 2024)
    print(df_is_yr.to_string(index=False))

    # OOS年次内訳（2025のみ）
    print("\n  === OOS 年次内訳（2025）===")
    df_oos_yr = yearly_table(res_oos, 2025, 2025)
    print(df_oos_yr.to_string(index=False))

    # ----------------------------------------------------------------
    # Step 5: チャート
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
    fig.suptitle(
        f"OOS検証 2025年  /  TEMPORAL24銘柄（2015-2017選定・固定）\n"
        f"IS: Sharpe={is_sharpe:.3f} CAGR={is_cagr:+.2f}%  |  "
        f"OOS: Sharpe={oos_sharpe:.3f} CAGR={oos_cagr:+.2f}%",
        fontsize=10,
    )

    # 上段: IS + OOS 連結資産曲線
    ax = axes[0]

    # IS曲線（2018-2024）
    ax.plot(eq_is.index, eq_is.values / 10_000, color="steelblue",
            linewidth=2.0, label=f"IS 2018-2024  Sharpe={is_sharpe:.3f}")

    # OOS曲線（2025、IS末尾に接続）— equity_curveは連続しているのでスケール不要
    ax.plot(eq_oos.index, eq_oos.values / 10_000,
            color="darkorange", linewidth=2.5, linestyle="-",
            label=f"OOS 2025  Sharpe={oos_sharpe:.3f}  CAGR={oos_cagr:+.2f}%")

    ax.axvline(pd.Timestamp("2025-01-01"), color="red", linestyle="--",
               linewidth=1.5, label="OOS境界")
    ax.axhline(CAPITAL / 10_000, color="black", linestyle="--",
               linewidth=0.7, label="初期資本")
    ax.set_ylabel("資産（万円）")
    ax.set_title("資産推移（青=IS / オレンジ=OOS）")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"¥{x:.0f}万"))

    # 下段: OOSのDD
    ax2 = axes[1]
    ax2.plot(dd_oos.index, dd_oos.values * 100, color="darkorange",
             linewidth=1.8, label=f"OOS MaxDD={oos_mdd:.2f}%")
    ax2.fill_between(dd_oos.index, dd_oos.values * 100, 0,
                     alpha=0.2, color="darkorange")
    ax2.axhline(-15, color="darkred", linestyle="--", linewidth=1.0, label="CB -15%")
    ax2.set_ylabel("ドローダウン（%）")
    ax2.set_title("OOS ドローダウン（2025）")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.tight_layout()
    out_png = os.path.join(OUTPUT, "oos_2025.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  チャート保存: {out_png}")

    # ----------------------------------------------------------------
    # JSON保存
    # ----------------------------------------------------------------
    result_json = {
        "run_date":    "2026-03-19",
        "universe":    UNIVERSE_JSON,
        "is_period":   f"{IS_START}/{IS_END}",
        "oos_period":  f"{OOS_START}/{OOS_END}",
        "IS": {
            "sharpe":        round(is_sharpe, 3),
            "cagr":          round(is_cagr, 2),
            "max_dd":        round(is_mdd, 2),
            "calmar":        round(is_calmar_v, 3),
            "avg_exposure":  round(is_exp, 3),
        },
        "OOS_2025": {
            "sharpe":        round(oos_sharpe, 3),
            "cagr":          round(oos_cagr, 2),
            "max_dd":        round(oos_mdd, 2),
            "calmar":        round(oos_calmar, 3),
            "avg_exposure":  round(oos_exp, 3),
            "trades_per_month": round(compute_trades_per_month(res_oos, OOS_START, OOS_END), 2),
            "days_to_recover":  round(compute_days_to_recover(res_oos), 1),
        },
        "diagnosis": {
            "phase1_ok":     bool(ph1_ok),
            "sharpe_range":  "ok" if 0.7 <= oos_sharpe <= 1.3 else
                             ("warn_low" if oos_sharpe < 0.7 else "warn_high"),
        },
    }
    out_json = "results/oos_2025_2026-03-19.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)
    print(f"  JSON保存: {out_json}")

    print("\n" + "=" * 72)
    print("  完了。")
    print(f"  OOS 要約: CAGR={oos_cagr:+.2f}%  Sharpe={oos_sharpe:.3f}  "
          f"MaxDD={oos_mdd:.2f}%  avg_exp={oos_exp:.3f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
