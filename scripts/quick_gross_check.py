"""
gross exposure 制御の効果を 2022年セグメント（2021-07-01〜2022-12-31）だけで素早く確認。
全WFより大幅に速く完了する。

実行:
    cd C:/ai-trading
    python scripts/quick_gross_check.py
"""
import sys
import os
import dataclasses

# C:/ai-trading を sys.path に追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.backtest.composite_alpha_bt import (
    run_scenario,
    _load_rsr_universe,
    _download_topix,
    _calc_regime,
    _precompute_tech_matrices,
    _calc_breadth,
    BREADTH_STOP,
    BREADTH_REDUCE,
    COMP_ALPHA_WINDOW,
)
from src.backtest.rsr import calc_universe_rsr, calc_composite_alpha_matrix
from src.backtest.universe_builder import download_universe
from src.strategy.universe import build_dyn_rsr42_active
from src.config_loader import load_strategy_config

SEG3_START = "2021-07-01"
SEG3_END   = "2022-12-31"

SCENARIO = "BASELINE"


def main() -> None:
    cfg = load_strategy_config()
    rc  = cfg.risk_controls

    print("=" * 60)
    print("  Gross Exposure 制御: 2022年セグメント比較")
    print(f"  期間: {SEG3_START} 〜 {SEG3_END}")
    print("=" * 60)

    # ---- データ準備 ----
    print("\n[1/5] ユニバース読み込み...")
    rsr_syms   = _load_rsr_universe(verbose=False)
    trade_syms = rsr_syms

    print(f"[2/5] 価格データ取得中（{len(rsr_syms)}銘柄）...")
    all_syms     = {**rsr_syms, **trade_syms}
    universe_raw = download_universe(
        all_syms,
        start=SEG3_START,
        end=SEG3_END,
        min_days=60,
        verbose=False,
    )
    topix_close = _download_topix(SEG3_START, SEG3_END, verbose=False)

    print(f"[3/5] RSR / Alpha 計算中...")
    rsr42_prices = {
        sym: universe_raw[sym]["df"]["Close"]
        for sym in rsr_syms if sym in universe_raw
    }
    rsr_df   = calc_universe_rsr(rsr42_prices)
    alpha_df = calc_composite_alpha_matrix(
        {sym: universe_raw[sym]["df"]["Close"] for sym in trade_syms if sym in universe_raw},
        window=COMP_ALPHA_WINDOW,
    )
    alpha_df = alpha_df.shift(1)   # 先読み防止

    regime_df = _calc_regime(topix_close)

    sym_active_df = build_dyn_rsr42_active(
        universe_raw=universe_raw,
        topix_close=topix_close,
        rsr_df=rsr_df,
        all_syms=list(trade_syms.keys()),
        start=SEG3_START,
        end=SEG3_END,
    )

    print("[4/5] テクニカル指標計算中...")
    tech_matrices  = _precompute_tech_matrices(universe_raw, list(trade_syms.keys()))
    breadth_series = _calc_breadth(universe_raw)

    # ---- パターン A: gross_exposure OFF ----
    print("\n[5/5] バックテスト実行中...")
    print("  パターン A (gross_exposure OFF)...")
    cfg_A = dataclasses.replace(
        cfg,
        risk_controls=dataclasses.replace(rc, gross_exposure_enabled=False),
    )
    res_A = run_scenario(
        scenario       = SCENARIO,
        universe_raw   = universe_raw,
        rsr_df         = rsr_df,
        alpha_df       = alpha_df,
        regime_df      = regime_df,
        trade_syms     = trade_syms,
        rsr_syms       = rsr_syms,
        cfg            = cfg_A,
        start          = SEG3_START,
        end            = SEG3_END,
        verbose        = False,
        tech_matrices  = tech_matrices,
        breadth_series = breadth_series,
        breadth_stop   = BREADTH_STOP,
        breadth_reduce = BREADTH_REDUCE,
        capital        = cfg.portfolio.capital,
        min_hold       = cfg.risk.min_hold_days,
        topix_close    = topix_close,
        sym_active_df  = sym_active_df,
    )

    # ---- パターン D: gross_exposure ON ----
    print("  パターン D (gross_exposure ON)...")
    cfg_D = dataclasses.replace(
        cfg,
        risk_controls=dataclasses.replace(rc, gross_exposure_enabled=True),
    )
    res_D = run_scenario(
        scenario       = SCENARIO,
        universe_raw   = universe_raw,
        rsr_df         = rsr_df,
        alpha_df       = alpha_df,
        regime_df      = regime_df,
        trade_syms     = trade_syms,
        rsr_syms       = rsr_syms,
        cfg            = cfg_D,
        start          = SEG3_START,
        end            = SEG3_END,
        verbose        = False,
        tech_matrices  = tech_matrices,
        breadth_series = breadth_series,
        breadth_stop   = BREADTH_STOP,
        breadth_reduce = BREADTH_REDUCE,
        capital        = cfg.portfolio.capital,
        min_hold       = cfg.risk.min_hold_days,
        topix_close    = topix_close,
        sym_active_df  = sym_active_df,
    )

    # ---- 結果集計 ----
    sk_A = res_A.get("skip_stats", {})
    sk_D = res_D.get("skip_stats", {})
    gc_days_D = res_D.get("gross_cap_active_days", 0)

    sharpe_A = res_A["sharpe"]
    maxdd_A  = res_A["max_dd"]        # 負値（例: -12.3）
    trades_A = res_A["n_trades"]
    skip_gross_A = sk_A.get("gross_exposure", 0)

    sharpe_D = res_D["sharpe"]
    maxdd_D  = res_D["max_dd"]
    trades_D = res_D["n_trades"]
    skip_gross_D = sk_D.get("gross_exposure", 0)

    total_signals_D = trades_D + skip_gross_D
    skip_rate_D = skip_gross_D / max(1, total_signals_D) * 100

    dd_improvement = maxdd_D - maxdd_A   # 正 = 改善（DD が浅くなった）
    sharpe_delta   = sharpe_D - sharpe_A  # 正 = 改善

    print()
    print("=" * 60)
    print("  Gross Exposure 制御: 2022年セグメント比較")
    print("=" * 60)
    print(f"パターン A (gross_exposure OFF): "
          f"Sharpe={sharpe_A:.3f}, MaxDD={maxdd_A:.1f}%, "
          f"trade={trades_A}, skip(gross)={skip_gross_A}")
    print(f"パターン D (gross_exposure ON):  "
          f"Sharpe={sharpe_D:.3f}, MaxDD={maxdd_D:.1f}%, "
          f"trade={trades_D}, skip(gross)={skip_gross_D}, "
          f"gross_cap発動日={gc_days_D}日")
    print(f"改善: MaxDD {dd_improvement:+.1f}pt, "
          f"Sharpe {'改善' if sharpe_delta >= 0 else '劣化'} {sharpe_delta:+.3f}")
    print()

    dd_ok     = dd_improvement >= 2.0
    sharpe_ok = sharpe_delta  >= -0.10
    skip_ok   = skip_rate_D   <= 15.0

    def mark(b: bool) -> str:
        return "✅" if b else "❌"

    print("採用判定:")
    print(f"  MaxDD改善 ≥ 2.0pt?  {mark(dd_ok)}  ({dd_improvement:+.1f}pt)")
    print(f"  Sharpe劣化 ≤ 0.10?  {mark(sharpe_ok)}  ({sharpe_delta:+.3f})")
    print(f"  skip率 ≤ 15%?       {mark(skip_ok)}  ({skip_rate_D:.1f}%)")

    adopted = dd_ok and sharpe_ok and skip_ok
    print()
    print(f"  → {'✅ 採用推奨' if adopted else '❌ 採用保留'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
