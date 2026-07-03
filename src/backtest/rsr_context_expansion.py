"""
backtest/rsr_context_expansion.py
RSR母集団拡張テスト: 42 → 62（42 live + 20 shadow）

【テスト条件】
  BASELINE  : RSR計算母集団 = 42 / 発注ユニバース = 42
  EXPANDED  : RSR計算母集団 = 62（42 + shadow20）/ 発注ユニバース = 42（変わらず）

【見る指標】
  trade_count / avg_exposure / sharpe / calmar / max_dd

【目的】
  - RSRランキングが62銘柄コンテキストで安定するか確認
  - trade_count +15〜25%、avg_exposure +5〜10% が期待値
  - 悪化（MaxDD拡大 / Sharpe低下）が起きないことを確認
"""

from __future__ import annotations
import os, sys, json, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from backtest.universe_builder import download_universe
from backtest.rsr              import calc_universe_rsr
from backtest.fujiko_strategy  import FujikoStrategy
from backtest.portfolio_engine import PortfolioEngine
from backtest.engine           import TradeCost

# ------------------------------------------------------------------ #
# 定数
# ------------------------------------------------------------------ #
START   = "2017-01-01"   # RSR計算に1年分のウォームアップが必要なため早めに取得
END     = "2024-12-31"
EVAL_START = "2018-01-01"  # 評価期間
CAPITAL = 2_000_000

# PortfolioEngine パラメータ（rsr42 baseline に合わせる）
PE_PARAMS = dict(
    capital          = CAPITAL,
    max_dd_limit     = 0.15,
    min_sectors      = 1,
    max_positions    = 3,
    cost             = TradeCost(),
    vol_target       = 0.0,         # 均等ウェイト
    max_single_weight= 0.15,        # rsr42 baseline と同じ
    vol_window       = 20,
    use_idm          = False,
    use_dd_scalar    = False,
)

MIN_RSR  = 70.0
MIN_SEPA = 6


# ------------------------------------------------------------------ #
# ユニバースファイル読み込み
# ------------------------------------------------------------------ #
def _load_universe_json(path: str) -> dict[str, str]:
    data = json.loads(open(path, encoding="utf-8").read())
    return data["symbols"]


def load_universes() -> tuple[dict[str, str], dict[str, str]]:
    base = os.path.dirname(os.path.dirname(__file__))
    trade_tickers  = _load_universe_json(
        os.path.join(base, "configs/universe/2026Q1_rsr42_universe.json"))
    shadow_tickers = _load_universe_json(
        os.path.join(base, "configs/universe/shadow_universe.json"))
    return trade_tickers, shadow_tickers


# ------------------------------------------------------------------ #
# 戦略辞書を組み立て（RSRは呼び出し元から渡す）
# ------------------------------------------------------------------ #
def build_strategies(
    trade_universe: dict,     # {sym: {"df": df, "sector": str}}
    rsr_uni:        pd.DataFrame,
    min_rsr:        float = MIN_RSR,
) -> dict:
    strat_dict = {}
    for sym, info in trade_universe.items():
        rsr_s = rsr_uni[sym] if sym in rsr_uni.columns else None
        strat_dict[sym] = FujikoStrategy(
            rsr_series    = rsr_s,
            min_sepa      = MIN_SEPA,
            min_rsr       = min_rsr,
            mom_period    = 21,
            turtle_entry  = 20,
            turtle_exit   = 20,
            use_turtle_entry = True,
        )
    return strat_dict


# ------------------------------------------------------------------ #
# 1シナリオ実行
# ------------------------------------------------------------------ #
def run_scenario(
    trade_universe: dict,
    strat_dict:     dict,
    label:          str,
) -> dict:
    res = PortfolioEngine(
        universe = trade_universe,
        strategy = strat_dict,
        **PE_PARAMS,
    ).run()

    # 評価期間だけ切り出す
    eq = res.equity_curve[res.equity_curve.index >= EVAL_START]
    dd = res.dd_series   [res.dd_series.index    >= EVAL_START]

    if eq.empty:
        return {"label": label, "error": "no data"}

    # CAGR
    n_years = len(eq) / 252
    cagr    = (eq.iloc[-1] / eq.iloc[0]) ** (1 / n_years) - 1 if n_years > 0 else 0
    max_dd  = float(dd.min())
    calmar  = cagr / abs(max_dd) if max_dd < 0 else float("inf")

    # Sharpe
    daily_ret = eq.pct_change().dropna()
    sharpe    = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0

    # トレード統計
    n_trades = res.n_trades
    avg_exp  = float(res.exposure_series.mean())

    return {
        "label":        label,
        "cagr_pct":     round(cagr * 100,    2),
        "max_dd_pct":   round(max_dd * 100,  2),
        "calmar":       round(calmar,         3),
        "sharpe":       round(sharpe,         3),
        "n_trades":     n_trades,
        "n_trades_per_year": round(n_trades / ((len(eq)) / 252), 1) if len(eq) > 0 else 0,
        "avg_exposure": round(avg_exp * 100,  1),
        "win_rate_pct": round(res.win_rate * 100, 1),
    }


# ------------------------------------------------------------------ #
# RSR turnover（ランキング安定性）計算
# ------------------------------------------------------------------ #
def calc_rsr_turnover(rsr_uni: pd.DataFrame, trade_syms: list[str]) -> float:
    """
    Top10 RSR ランキングの1日あたり入れ替わり率（0〜1）。
    低いほどランキングが安定。
    """
    sub = rsr_uni[[s for s in trade_syms if s in rsr_uni.columns]]
    k   = min(10, len(sub.columns))
    tops = sub.rank(axis=1, ascending=False) <= k
    changes = tops.astype(int).diff().abs().sum(axis=1)
    return float(changes.mean() / (2 * k)) if k > 0 else 0.0


# ------------------------------------------------------------------ #
# MAIN
# ------------------------------------------------------------------ #
def main() -> None:
    print("=" * 68)
    print("  RSR母集団拡張テスト: 42 → 62（42 live + 20 shadow）")
    print(f"  評価期間: {EVAL_START} / {END}  初期資本: ¥{CAPITAL:,}")
    print("=" * 68)

    # --- 1. ユニバース読み込み ---
    trade_tickers, shadow_tickers = load_universes()
    all_tickers = {**trade_tickers, **shadow_tickers}

    print(f"\n[1/4] データ取得中...")
    print(f"  発注ユニバース: {len(trade_tickers)}銘柄")
    print(f"  Shadowユニバース: {len(shadow_tickers)}銘柄")
    print(f"  RSR拡張後合計: {len(all_tickers)}銘柄")

    # 全銘柄ダウンロード（重複なし）
    all_data = download_universe(all_tickers, start=START, end=END, verbose=False)
    print(f"  取得完了: {len(all_data)} / {len(all_tickers)} 銘柄")

    # 発注ユニバースだけ抽出
    trade_data   = {s: all_data[s] for s in trade_tickers  if s in all_data}
    shadow_data  = {s: all_data[s] for s in shadow_tickers if s in all_data}

    print(f"  発注ユニバース有効: {len(trade_data)}銘柄")
    print(f"  Shadow有効: {len(shadow_data)}銘柄")

    # --- 2. RSR計算（2パターン） ---
    print("\n[2/4] RSR計算...")

    prices_42 = {s: v["df"]["Close"] for s, v in trade_data.items()}
    prices_62 = {s: v["df"]["Close"] for s, v in all_data.items()}

    rsr_42 = calc_universe_rsr(prices_42)  # BASELINE: 42コンテキスト
    rsr_62 = calc_universe_rsr(prices_62)  # EXPANDED: 62コンテキスト

    print(f"  BASELINE RSR: {rsr_42.shape[1]}銘柄コンテキスト")
    print(f"  EXPANDED RSR: {rsr_62.shape[1]}銘柄コンテキスト")

    # RSR turnover（ランキング安定性）
    trade_syms = list(trade_data.keys())
    turnover_42 = calc_rsr_turnover(rsr_42, trade_syms)
    turnover_62 = calc_rsr_turnover(rsr_62, trade_syms)
    print(f"\n  RSR Turnover（Top10入れ替え率/日）:")
    print(f"    BASELINE (42): {turnover_42:.3f}")
    print(f"    EXPANDED (62): {turnover_62:.3f}")
    print(f"    {'↓ 安定化' if turnover_62 < turnover_42 else '↑ 変動増'} (差: {turnover_62 - turnover_42:+.3f})")

    # RSR平均値の比較（発注銘柄42本）
    latest_42 = rsr_42.iloc[-1][trade_syms]
    latest_62 = rsr_62.iloc[-1][[s for s in trade_syms if s in rsr_62.columns]]
    print(f"\n  最新RSR（発注42銘柄、評価期間末）:")
    print(f"    BASELINE (42): mean={latest_42.mean():.1f}  top10>{latest_42.nlargest(10).min():.1f}")
    print(f"    EXPANDED (62): mean={latest_62.mean():.1f}  top10>{latest_62.nlargest(10).min():.1f}")

    # --- 3. バックテスト実行 ---
    print("\n[3/4] バックテスト実行...")

    print("  [BASELINE] RSR42コンテキスト...")
    strat_42 = build_strategies(trade_data, rsr_42)
    result_42 = run_scenario(trade_data, strat_42, "BASELINE (RSR42)")
    print(f"    CAGR={result_42['cagr_pct']}%  MaxDD={result_42['max_dd_pct']}%  "
          f"Sharpe={result_42['sharpe']}  n_trades={result_42['n_trades']}  "
          f"exp={result_42['avg_exposure']}%")

    print("  [EXPANDED] RSR62コンテキスト（発注は42のまま）...")
    strat_62 = build_strategies(trade_data, rsr_62)
    result_62 = run_scenario(trade_data, strat_62, "EXPANDED (RSR62)")
    print(f"    CAGR={result_62['cagr_pct']}%  MaxDD={result_62['max_dd_pct']}%  "
          f"Sharpe={result_62['sharpe']}  n_trades={result_62['n_trades']}  "
          f"exp={result_62['avg_exposure']}%")

    # --- 4. 比較サマリー ---
    print("\n" + "=" * 68)
    print("  比較結果サマリー")
    print("=" * 68)

    cols = ["label", "cagr_pct", "max_dd_pct", "calmar", "sharpe",
            "n_trades", "n_trades_per_year", "avg_exposure", "win_rate_pct"]
    df_cmp = pd.DataFrame([result_42, result_62])[cols]
    df_cmp.columns = ["シナリオ", "CAGR%", "MaxDD%", "Calmar", "Sharpe",
                      "取引数", "取引/年", "露出%", "勝率%"]
    print(df_cmp.to_string(index=False))

    # 変化量
    print("\n  === EXPANDED vs BASELINE 差分 ===")
    for key, label in [
        ("n_trades",        "取引数"),
        ("n_trades_per_year","取引/年"),
        ("avg_exposure",    "露出%"),
        ("cagr_pct",        "CAGR%"),
        ("max_dd_pct",      "MaxDD%"),
        ("sharpe",          "Sharpe"),
        ("calmar",          "Calmar"),
    ]:
        v42 = result_42[key]
        v62 = result_62[key]
        diff = v62 - v42
        pct  = diff / abs(v42) * 100 if v42 != 0 else float("inf")
        sign = "+" if diff > 0 else ""
        print(f"  {label:<12}: {v42} → {v62}  ({sign}{diff:.1f},  {sign}{pct:.1f}%)")

    print(f"\n  RSR Turnover安定性: {turnover_42:.3f} → {turnover_62:.3f}")

    # 判定
    trade_up  = result_62["n_trades"]     > result_42["n_trades"]
    exp_up    = result_62["avg_exposure"] > result_42["avg_exposure"]
    dd_ok     = abs(result_62["max_dd_pct"]) <= abs(result_42["max_dd_pct"]) * 1.15
    sharpe_ok = result_62["sharpe"]       >= result_42["sharpe"] * 0.85

    print("\n  === 合格判定 ===")
    print(f"  取引数増加:   {'✅' if trade_up else '❌'} ({result_42['n_trades']} → {result_62['n_trades']})")
    print(f"  露出増加:     {'✅' if exp_up else '❌'} ({result_42['avg_exposure']}% → {result_62['avg_exposure']}%)")
    print(f"  MaxDD許容内:  {'✅' if dd_ok else '❌'} ({result_42['max_dd_pct']}% → {result_62['max_dd_pct']}%  ±15%以内)")
    print(f"  Sharpe許容内: {'✅' if sharpe_ok else '❌'} ({result_42['sharpe']} → {result_62['sharpe']}  -15%以内)")

    passed = trade_up and exp_up and dd_ok and sharpe_ok
    print(f"\n  【総合判定】 {'✅ PASS — RSR62コンテキスト採用推奨' if passed else '❌ FAIL — 採用見送り'}")

    # --- 5. 結果保存 ---
    import datetime
    today = datetime.date.today().isoformat()
    output = {
        "run_date":    today,
        "eval_period": f"{EVAL_START} / {END}",
        "baseline": result_42,
        "expanded": result_62,
        "rsr_turnover": {
            "baseline_42": round(turnover_42, 4),
            "expanded_62": round(turnover_62, 4),
            "delta": round(turnover_62 - turnover_42, 4),
        },
        "verdict": "PASS" if passed else "FAIL",
    }
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "results", f"rsr_context_expansion_{today}.json"
    )
    import json as _json
    with open(out_path, "w", encoding="utf-8") as f:
        _json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n[4/4] 結果保存: {out_path}")


if __name__ == "__main__":
    main()
