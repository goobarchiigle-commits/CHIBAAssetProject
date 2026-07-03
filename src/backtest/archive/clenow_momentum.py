"""
backtest/clenow_momentum.py
Clenow Momentum（slope × R²）vs RSR フィルター 比較バックテスト

【Clenow 指標の定義】
  - 過去 N 日間の log(price) を時間 t に対して線形回帰
  - slope: 回帰係数（日次上昇率 ≈ exp(slope) - 1）
  - R²:    回帰の当てはまり度（滑らかなトレンドほど高い）
  - Clenow スコア = slope × R² × 10000  （スケーリングで人間が読める値に）

【RSR との違い】
  RSR:     相対強度ランキング（ベンチマーク比の過去パフォーマンス）
  Clenow:  トレンドの「滑らかさ」× 「傾き」→ 急騰1発より月次コツコツ上昇を好む

【比較シナリオ】
  A: RSR フィルターのみ（min_rsr=70、現行設定）
  B: RSR + Clenow フィルター（min_rsr=70 AND clenow_score > 0）
  C: Clenow のみでランキング・フィルター（RSR 無効、clenow_score>0 の上位 k 銘柄）

【使い方】
  python -m backtest.clenow_momentum
"""

from __future__ import annotations
import os, sys, warnings, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from backtest.universe_builder        import download_universe
from backtest.rsr                     import calc_universe_rsr
from backtest.fujiko_strategy         import FujikoStrategy
from backtest.mean_reversion_strategy import MeanReversionStrategy
from backtest.portfolio_engine        import PortfolioEngine
from backtest.engine                  import TradeCost

START   = "2018-01-01"
END     = "2024-12-31"
CAPITAL = 2_000_000
CLENOW_PERIOD = 90   # Clenow は 90 日回帰が標準

UNIVERSE_FILE = "configs/universe/2026Q1_temporal24.json"

SECTOR_STRATEGY: dict[str, str] = {
    "海運":"fujiko","機械":"fujiko","電機精密":"fujiko","商社":"fujiko",
    "電機":"fujiko","食品":"fujiko","情報通信":"dynamic","小売":"dynamic",
    "医薬品":"dynamic","不動産":"dynamic","サービス":"dynamic","レジャー":"fujiko",
    "ガス":"mean_rev","鉄鋼":"mean_rev","銀行":"mean_rev","保険":"mean_rev",
    "輸送機器":"mean_rev","化学":"mean_rev","陸運":"mean_rev","石油":"mean_rev",
}
MR_PARAMS = dict(
    rsi_period=5, rsi_entry=25.0, rsi_exit=65.0,
    ma_long=200, stop_loss_pct=0.07, max_hold_days=10, knife_threshold=0.15,
)


# ------------------------------------------------------------------ #
# Clenow Momentum 計算
# ------------------------------------------------------------------ #
def calc_clenow_score(prices: pd.Series, period: int = CLENOW_PERIOD) -> float:
    """
    Clenow スコア = slope × R² × 10000

    Args:
        prices: 終値 Series（期間は十分に長いこと）
        period: 回帰期間（デフォルト 90 日）

    Returns:
        float（大きいほど「滑らかに上昇中」。0未満 = 下落トレンド）
    """
    if len(prices) < period:
        return float("nan")

    y = np.log(prices.values[-period:].astype(float))
    if np.any(y <= 0) or np.any(np.isnan(y)):
        return float("nan")

    x     = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, 1)

    y_pred = slope * x + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    r2     = max(0.0, min(1.0, r2))

    return slope * r2 * 10_000   # スケーリング（annualized pct × R² × 100 に近い）


def calc_universe_clenow(
    prices: dict[str, pd.Series],
    period: int = CLENOW_PERIOD,
) -> pd.DataFrame:
    """
    全銘柄の Clenow スコアを日次で計算する。

    Returns:
        DataFrame（index=date, columns=symbol）
    """
    records: dict[str, pd.Series] = {}
    for sym, ps in prices.items():
        ps = ps.dropna()
        scores = []
        idx    = []
        for i in range(period, len(ps)):
            score = calc_clenow_score(ps.iloc[:i + 1], period=period)
            scores.append(score)
            idx.append(ps.index[i])
        records[sym] = pd.Series(scores, index=idx)

    return pd.DataFrame(records)


# ------------------------------------------------------------------ #
# Clenow フィルター付き FujikoStrategy ラッパー
# ------------------------------------------------------------------ #
class ClenowFilteredStrategy:
    """
    FujikoStrategy に Clenow スコア > 0 フィルターを追加したラッパー。
    BUY シグナルが出ても Clenow スコアが 0 以下なら HOLD に変換する。
    """

    def __init__(
        self,
        base_strategy,
        clenow_series:  pd.Series | None,
        min_clenow:     float = 0.0,
    ):
        self._base         = base_strategy
        self._clenow       = clenow_series
        self._min_clenow   = min_clenow

    @property
    def name(self) -> str:
        return self._base.name + "+Clenow"

    def generate_signal(self, data: pd.DataFrame) -> int:
        signal = self._base.generate_signal(data)
        if signal == 1 and self._clenow is not None:
            date    = data.index[-1]
            past_c  = self._clenow.loc[:date]
            if len(past_c) == 0:
                return 0   # スコアがまだ計算できていない期間はスキップ
            score = float(past_c.iloc[-1])
            if np.isnan(score) or score <= self._min_clenow:
                return 0   # Clenow フィルター不通過
        return signal


# ------------------------------------------------------------------ #
# ヘルパー
# ------------------------------------------------------------------ #
def calmar(cagr: float, maxdd: float) -> float:
    return cagr / abs(maxdd) if maxdd != 0 else float("inf")


def build_strats(
    universe:   dict,
    rsr_uni:    pd.DataFrame,
    clenow_df:  pd.DataFrame | None,
    use_clenow: bool,
    min_rsr:    float,
) -> dict:
    strats = {}
    for sym, info in universe.items():
        rsr_s  = rsr_uni[sym] if sym in rsr_uni.columns else None
        sector = info["sector"]
        rule   = SECTOR_STRATEGY.get(sector, "dynamic")

        if rule == "mean_rev":
            base = MeanReversionStrategy(**MR_PARAMS)
        else:
            base = FujikoStrategy(
                rsr_series=rsr_s, min_sepa=6, min_rsr=min_rsr,
                mom_period=21, turtle_entry=20, turtle_exit=20,
                use_turtle_entry=True,
            )

        if use_clenow and clenow_df is not None and sym in clenow_df.columns:
            strats[sym] = ClenowFilteredStrategy(base, clenow_df[sym])
        else:
            strats[sym] = base

    return strats


def run_scenario(universe, strats) -> object:
    return PortfolioEngine(
        universe=universe, strategy=strats, capital=CAPITAL,
        max_dd_limit=0.15, min_sectors=1, max_positions=3,
        cost=TradeCost(), vol_target=0.0, max_single_weight=0.25,
        vol_window=20, use_idm=False, use_dd_scalar=False,
    ).run()


def metrics(res, label: str) -> dict:
    cagr  = res.cagr
    maxdd = res.max_drawdown
    return {
        "シナリオ":  label,
        "CAGR%":    round(cagr * 100, 2),
        "MaxDD%":   round(maxdd * 100, 2),
        "Calmar":   round(calmar(cagr, maxdd), 3),
        "Sharpe":   round(res.sharpe_ratio, 3),
        "取引数":   res.n_trades,
        "勝率%":    round(res.win_rate * 100, 1),
        "avg_pos":  round(float(res.n_positions.mean()), 2),
    }


# ================================================================
# MAIN
# ================================================================
def main():
    print("=" * 66)
    print("  Clenow Momentum vs RSR フィルター 比較バックテスト")
    print(f"  期間: {START} 〜 {END}  /  Clenow期間: {CLENOW_PERIOD}日")
    print("=" * 66)

    # ---- 1. ユニバース ----
    with open(UNIVERSE_FILE, encoding="utf-8") as f:
        univ_cfg = json.load(f)
    tickers: dict[str, str] = univ_cfg["symbols"]

    print(f"\n[1/5] データ取得中（{len(tickers)}銘柄）...")
    universe = download_universe(tickers, start=START, end=END, verbose=False)
    print(f"  取得完了: {len(universe)} 銘柄")

    # ---- 2. RSR + Clenow 計算 ----
    prices_dict = {sym: info["df"]["Close"] for sym, info in universe.items()}
    rsr         = calc_universe_rsr(prices_dict)
    print(f"  RSR計算完了: {rsr.shape[1]} 銘柄")

    print(f"  Clenow スコア計算中（{CLENOW_PERIOD}日回帰）...")
    clenow_df = calc_universe_clenow(prices_dict, period=CLENOW_PERIOD)
    print(f"  Clenow計算完了: {clenow_df.shape[1]} 銘柄")

    # Clenow スコアの統計（参考）
    latest_c = clenow_df.iloc[-1].dropna()
    pos_pct  = (latest_c > 0).mean()
    print(f"  直近 Clenow > 0 の銘柄: {pos_pct:.0%}  "
          f"中央値={latest_c.median():.2f}  平均={latest_c.mean():.2f}")

    # ---- 3. シナリオ実行 ----
    print("\n[2/5] シナリオ A: RSRのみ（min_rsr=70）...")
    sa    = build_strats(universe, rsr, None,      use_clenow=False, min_rsr=70.0)
    res_a = run_scenario(universe, sa)
    m_a   = metrics(res_a, "A: RSRのみ")
    print(f"  CAGR={m_a['CAGR%']:+.2f}%  MaxDD={m_a['MaxDD%']:.2f}%  "
          f"Calmar={m_a['Calmar']:.3f}  Sharpe={m_a['Sharpe']:.3f}  取引={m_a['取引数']}")

    print("\n[3/5] シナリオ B: RSR + Clenow フィルター（両方必須）...")
    sb    = build_strats(universe, rsr, clenow_df, use_clenow=True,  min_rsr=70.0)
    res_b = run_scenario(universe, sb)
    m_b   = metrics(res_b, "B: RSR+Clenow")
    print(f"  CAGR={m_b['CAGR%']:+.2f}%  MaxDD={m_b['MaxDD%']:.2f}%  "
          f"Calmar={m_b['Calmar']:.3f}  Sharpe={m_b['Sharpe']:.3f}  取引={m_b['取引数']}")

    print("\n[4/5] シナリオ C: Clenowのみ（RSR 無効、clenow>0 のみ）...")
    sc    = build_strats(universe, rsr, clenow_df, use_clenow=True,  min_rsr=0.0)
    res_c = run_scenario(universe, sc)
    m_c   = metrics(res_c, "C: Clenowのみ")
    print(f"  CAGR={m_c['CAGR%']:+.2f}%  MaxDD={m_c['MaxDD%']:.2f}%  "
          f"Calmar={m_c['Calmar']:.3f}  Sharpe={m_c['Sharpe']:.3f}  取引={m_c['取引数']}")

    # ---- 4. 比較表 ----
    print("\n[5/5] 結果比較")
    SEP = "=" * 72
    print(SEP)
    keys = ["CAGR%","MaxDD%","Calmar","Sharpe","取引数","勝率%","avg_pos"]
    header = f"  {'指標':<12} {'A: RSRのみ':>14} {'B: RSR+Clenow':>14} {'C: Clenowのみ':>14}"
    print(header)
    print("-" * 72)
    for k in keys:
        print(f"  {k:<12} {str(m_a[k]):>14} {str(m_b[k]):>14} {str(m_c[k]):>14}")
    print(SEP)

    # 判定
    results_calmar = {"A": m_a["Calmar"], "B": m_b["Calmar"], "C": m_c["Calmar"]}
    best = max(results_calmar, key=results_calmar.get)
    best_val = results_calmar[best]

    verdicts = {
        "A": "RSRのみ（現行維持）",
        "B": "RSR + Clenow フィルター（両方必須）→ 採用推奨",
        "C": "Clenow のみ（RSR廃止）",
    }
    print(f"\n  Calmar ベスト: {best}（{best_val:.3f}）→ {verdicts[best]}")

    # Clenow の効果判定
    if m_b["Calmar"] > m_a["Calmar"]:
        effect = f"✅ Clenow フィルター有効: Calmar {m_a['Calmar']:.3f}→{m_b['Calmar']:.3f}"
    else:
        effect = f"→ Clenow フィルター効果なし: Calmar {m_a['Calmar']:.3f}→{m_b['Calmar']:.3f}"
    print(f"  {effect}")

    if m_b["取引数"] < m_a["取引数"] * 0.7:
        print(f"  ⚠ 取引数が大幅減少（{m_a['取引数']}→{m_b['取引数']}）: フィルター強すぎ疑い")

    print()

    # JSON 保存
    out = {
        "generated_at": pd.Timestamp.now().isoformat()[:19],
        "period":       f"{START}/{END}",
        "clenow_period": CLENOW_PERIOD,
        "A_rsr_only":    {k: v for k, v in m_a.items() if k != "シナリオ"},
        "B_rsr_clenow":  {k: v for k, v in m_b.items() if k != "シナリオ"},
        "C_clenow_only": {k: v for k, v in m_c.items() if k != "シナリオ"},
        "best":          best,
        "effect":        effect,
    }
    os.makedirs("results", exist_ok=True)
    with open("results/clenow_comparison.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("  結果保存: results/clenow_comparison.json")


if __name__ == "__main__":
    main()
