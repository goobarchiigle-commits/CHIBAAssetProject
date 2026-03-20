"""
backtest/portfolio_atr_weight.py
A/B バックテスト: 均等配分 vs ATR逆数配分（VolatilityParity）

【比較シナリオ】
  A: vol_target=0   → 均等配分（現在の V2 設定と同等）
  B: vol_target=0.02 → ATR 逆数配分（低ボラ銘柄に多く配分）

【追加指標】
  position_size_variance: ポジション間のサイズ分散（np.var(weights)）
  avg_exposure: 平均投資比率（invested / equity）

【使い方】
  python -m backtest.portfolio_atr_weight
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
from backtest.portfolio_engine        import PortfolioEngine, Position, _IDM_TABLE
from backtest.engine                  import TradeCost
from portfolio.volatility_allocator   import calc_atr

START   = "2018-01-01"
END     = "2024-12-31"
CAPITAL = 2_000_000

# ------------------------------------------------------------------ #
# ユニバース（temporal24 — バイアスなし）
# ------------------------------------------------------------------ #
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
FUJIKO_PARAMS = dict(
    min_sepa=6, min_rsr=70.0, mom_period=21,
    turtle_entry=20, turtle_exit=20, use_turtle_entry=True,
)


# ------------------------------------------------------------------ #
# ATR加重エンジン（PortfolioEngine をサブクラス化）
# ------------------------------------------------------------------ #
class ATRWeightedEngine(PortfolioEngine):
    """
    `_vol_adjusted_alloc` を ATR ベースに差し替えたエンジン。
    他のロジック（シグナル/CB/セクター/DD）はすべて親クラスそのまま。
    """

    def _vol_adjusted_alloc(
        self,
        symbol:    str,
        past_df:   pd.DataFrame,
        n_current: int = None,
    ) -> float:
        """ATR 逆数サイジング: target_risk * capital / ATR_pct"""
        atr = calc_atr(past_df, period=self.vol_window)
        if np.isnan(atr) or atr <= 0:
            return self._position_alloc()

        price = float(past_df["Close"].iloc[-1])
        if price <= 0:
            return self._position_alloc()

        atr_pct = atr / price                      # 価格正規化 ATR（ボラ率）

        if self.use_idm and n_current is not None and n_current >= 1:
            idm = _IDM_TABLE.get(n_current, 1.48)
            target_risk = self.vol_target * idm / n_current
        else:
            target_risk = self.vol_target / self.max_positions

        alloc = target_risk * self.capital / atr_pct

        alloc = min(alloc, self.capital * self.max_single_weight)
        alloc = max(alloc, self._position_alloc() * 0.5)
        return alloc


# ------------------------------------------------------------------ #
# ヘルパー
# ------------------------------------------------------------------ #
def calmar(cagr: float, maxdd: float) -> float:
    return cagr / abs(maxdd) if maxdd != 0 else float("inf")


def build_strats(universe: dict, rsr_uni: pd.DataFrame) -> dict:
    """セクター別戦略辞書を構築する"""
    strats = {}
    for sym, info in universe.items():
        rsr_s  = rsr_uni[sym] if sym in rsr_uni.columns else None
        sector = info["sector"]
        rule   = SECTOR_STRATEGY.get(sector, "dynamic")
        if rule == "mean_rev":
            strats[sym] = MeanReversionStrategy(**MR_PARAMS)
        else:
            strats[sym] = FujikoStrategy(rsr_series=rsr_s, **FUJIKO_PARAMS)
    return strats


def calc_exposure_series(res) -> pd.Series:
    """equity_curve と n_positions から投資比率を推計"""
    n   = res.n_positions
    eq  = res.equity_curve
    # トレード記録から invested_capital を再現（近似）
    if not res.trades.empty:
        buys  = res.trades[res.trades["side"] == "BUY"]
        sells = res.trades[res.trades["side"] == "SELL"]
        invested = pd.Series(0.0, index=eq.index)
        for _, r in buys.iterrows():
            invested.loc[r["date"]:] += r["qty"] * r["price"]
        for _, r in sells.iterrows():
            invested.loc[r["date"]:] -= r["qty"] * r["price"]
        invested = invested.clip(0)
    else:
        # フォールバック: 均等配分仮定
        invested = n / 3 * (CAPITAL * 0.25)
    return (invested / eq).clip(0, 1)


def calc_position_variance(res) -> float:
    """ポジション間のウェイト分散（均等配分なら 0 に近い）"""
    if res.trades.empty:
        return 0.0
    buys = res.trades[res.trades["side"] == "BUY"].copy()
    if buys.empty:
        return 0.0
    buys["amt"]    = buys["qty"] * buys["price"]
    buys["weight"] = buys["amt"] / CAPITAL
    return float(np.var(buys["weight"]))


def metrics(res, label: str) -> dict:
    cagr  = res.cagr
    maxdd = res.max_drawdown
    exp   = calc_exposure_series(res)
    return {
        "シナリオ":            label,
        "CAGR%":              round(cagr * 100, 2),
        "MaxDD%":             round(maxdd * 100, 2),
        "Calmar":             round(calmar(cagr, maxdd), 3),
        "Sharpe":             round(res.sharpe_ratio, 3),
        "avg_exposure":       round(float(exp.mean()), 3),
        "position_size_var":  round(calc_position_variance(res), 5),
        "avg_positions":      round(float(res.n_positions.mean()), 2),
        "n_trades":           res.n_trades,
        "win_rate%":          round(res.win_rate * 100, 1),
    }


def run_engine(universe, strats, use_atr: bool, vol_tgt: float) -> object:
    cls = ATRWeightedEngine if use_atr else PortfolioEngine
    return cls(
        universe          = universe,
        strategy          = strats,
        capital           = CAPITAL,
        max_dd_limit      = 0.15,
        min_sectors       = 1,
        max_positions     = 3,
        cost              = TradeCost(),
        vol_target        = vol_tgt,
        max_single_weight = 0.25,
        vol_window        = 20,
        use_idm           = False,
        use_dd_scalar     = False,
    ).run()


# ================================================================
# MAIN
# ================================================================
def main():
    print("=" * 66)
    print("  A/B バックテスト: 均等配分 vs ATR逆数配分")
    print(f"  期間: {START} 〜 {END}  /  初期資本: ¥{CAPITAL:,}")
    print("=" * 66)

    # ---- 1. ユニバース読み込み ----
    with open(UNIVERSE_FILE, encoding="utf-8") as f:
        univ_cfg = json.load(f)
    tickers: dict[str, str] = univ_cfg["symbols"]

    print(f"\n[1/4] データ取得中（{len(tickers)}銘柄）...")
    universe = download_universe(tickers, start=START, end=END, verbose=False)
    print(f"  取得完了: {len(universe)} 銘柄")

    # ---- 2. RSR 計算 ----
    prices = {sym: info["df"]["Close"] for sym, info in universe.items()}
    rsr    = calc_universe_rsr(prices)
    print(f"  RSR計算完了: {rsr.shape[1]} 銘柄")

    # ---- 3. 戦略辞書 ----
    strats = build_strats(universe, rsr)

    # ---- 4. A/B 実行 ----
    print("\n[2/4] シナリオ A: 均等配分（vol_target=0）...")
    res_a  = run_engine(universe, strats, use_atr=False, vol_tgt=0.0)
    m_a    = metrics(res_a, "A: 均等配分")
    print(f"  CAGR={m_a['CAGR%']:+.2f}%  MaxDD={m_a['MaxDD%']:.2f}%  "
          f"Calmar={m_a['Calmar']:.3f}  Sharpe={m_a['Sharpe']:.3f}")

    print("\n[3/4] シナリオ B: ATR逆数配分（vol_target=0.02）...")
    res_b  = run_engine(universe, strats, use_atr=True,  vol_tgt=0.02)
    m_b    = metrics(res_b, "B: ATR逆数配分")
    print(f"  CAGR={m_b['CAGR%']:+.2f}%  MaxDD={m_b['MaxDD%']:.2f}%  "
          f"Calmar={m_b['Calmar']:.3f}  Sharpe={m_b['Sharpe']:.3f}")

    # ---- 5. 比較表 ----
    print("\n[4/4] 結果比較")
    SEP = "=" * 66
    print(SEP)
    header = (f"{'指標':<22} {'A: 均等配分':>14} {'B: ATR逆数':>14} {'差(B-A)':>10}")
    print(header)
    print("-" * 66)

    compare_keys = [
        ("CAGR%",            "%",    False),
        ("MaxDD%",           "%",    True),   # 小さい方が良いので負差が改善
        ("Calmar",           "",     False),
        ("Sharpe",           "",     False),
        ("avg_exposure",     "",     False),
        ("position_size_var","",     True),   # 小さい方が均等
        ("avg_positions",    "銘柄", False),
        ("n_trades",         "回",   False),
        ("win_rate%",        "%",    False),
    ]
    for key, unit, lower_is_better in compare_keys:
        va  = m_a[key]
        vb  = m_b[key]
        try:
            diff = float(vb) - float(va)
            if lower_is_better:
                mark = "↑" if diff < 0 else ("↓" if diff > 0 else "→")
            else:
                mark = "↑" if diff > 0 else ("↓" if diff < 0 else "→")
            diff_str = f"{diff:+.3f}{unit} {mark}"
        except (TypeError, ValueError):
            diff_str = "N/A"
        print(f"  {key:<20} {str(va)+unit:>14} {str(vb)+unit:>14} {diff_str:>12}")

    print(SEP)

    # 総合判定
    if m_b["Calmar"] > m_a["Calmar"] and abs(m_b["MaxDD%"]) < abs(m_a["MaxDD%"]):
        verdict = "✅ B(ATR逆数配分) 採用推奨: Calmar↑ MaxDD↓"
    elif m_b["Calmar"] > m_a["Calmar"]:
        verdict = "✅ B(ATR逆数配分) 採用推奨: Calmar↑"
    elif abs(m_b["MaxDD%"]) < abs(m_a["MaxDD%"]) and m_b["CAGR%"] >= m_a["CAGR%"] * 0.95:
        verdict = "✅ B(ATR逆数配分) 採用推奨: MaxDD↓ CAGR許容範囲内"
    else:
        verdict = "→ A(均等配分) 継続: ATR逆数配分による改善なし"

    print(f"\n  判定: {verdict}")
    print()

    # ---- 6. JSON 保存 ----
    out = {
        "generated_at": pd.Timestamp.now().isoformat()[:19],
        "period":        f"{START}/{END}",
        "A_equal":       {k: v for k, v in m_a.items() if k != "シナリオ"},
        "B_atr":         {k: v for k, v in m_b.items() if k != "シナリオ"},
        "verdict":       verdict,
    }
    out_path = "results/atr_weight_ab_test.json"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"  結果保存: {out_path}")


if __name__ == "__main__":
    main()
