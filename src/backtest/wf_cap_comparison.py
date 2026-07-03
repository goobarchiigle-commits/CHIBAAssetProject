"""
backtest/wf_cap_comparison.py

Step 1/2/3: dynamic_cap feature flag + cluster cap + bear adaptive cap WF比較
3パターンを WF 全5セグメント + true OOS 2025 で比較する。

パターン:
  A: 固定 cap のみ（symbol 8%, sector 25%）
  B: 固定 cap + cluster cap 35%
  C: 固定 cap + cluster cap + Bear adaptive（sector 18%, cluster 25%）

採用基準:
  B: 2022 MaxDD 改善 >= 2.0pt  AND  2025 Sharpe 劣化 <= 0.10
  C: 同上

実行:
  cd C:/ai-trading
  python src/backtest/wf_cap_comparison.py
"""

from __future__ import annotations
import os, sys, json, warnings, time, copy
from dataclasses import replace
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import backtest.composite_alpha_bt as _bt
from backtest.wf_dynamic_universe import (
    load_data, _build_active,
    BULL_ACTIVE_N, BEAR_ACTIVE_N,
    WF_SEGS, TRUE_OOS, IS_FULL, OUTPUT_DIR
)
from src.config_loader import load_strategy_config, RiskControlsConfig, StrategyConfig

os.environ.pop("DATA_VERSION", None)

OUTPUT_JSON = os.path.join(OUTPUT_DIR, f"wf_cap_comparison_{time.strftime('%Y-%m-%d')}.json")

SHOCK_MODE = "composite"


# ------------------------------------------------------------------ #
# cfg を cap 設定で上書きして新しい StrategyConfig を作成
# ------------------------------------------------------------------ #
def _make_cfg(base_cfg, cluster_cap: float, bear_sector_cap: float, bear_cluster_cap: float):
    """base_cfg の risk_controls を差し替えた新しい StrategyConfig を返す。"""
    new_rc = RiskControlsConfig(
        shock_exit_mode  = base_cfg.risk_controls.shock_exit_mode,
        regime_sizing    = base_cfg.risk_controls.regime_sizing,
        bear_scale       = base_cfg.risk_controls.bear_scale,
        dynamic_cap      = False,               # production 固定
        symbol_cap       = base_cfg.risk_controls.symbol_cap,
        sector_cap       = base_cfg.risk_controls.sector_cap,
        cluster_cap      = cluster_cap,
        bear_sector_cap  = bear_sector_cap,
        bear_cluster_cap = bear_cluster_cap,
    )
    return StrategyConfig(
        fujiko         = base_cfg.fujiko,
        mean_reversion = base_cfg.mean_reversion,
        portfolio      = base_cfg.portfolio,
        risk           = base_cfg.risk,
        risk_controls  = new_rc,
    )


# ------------------------------------------------------------------ #
# 1セグメント実行
# ------------------------------------------------------------------ #
def run_seg(universe_raw, rsr_df, alpha_df, regime_df, all_syms,
            tech_matrices, topix_close, cfg,
            start: str, end: str,
            sym_active_df=None) -> dict:
    return _bt.run_scenario(
        scenario                 = "BASELINE",
        universe_raw             = universe_raw,
        rsr_df                   = rsr_df,
        alpha_df                 = alpha_df,
        regime_df                = regime_df,
        trade_syms               = all_syms,
        rsr_syms                 = all_syms,
        cfg                      = cfg,
        start                    = start,
        end                      = end,
        capital                  = float(cfg.portfolio.capital),
        verbose                  = False,
        tech_matrices            = tech_matrices,
        topix_close              = topix_close,
        min_hold                 = int(cfg.risk.min_hold_days),
        market_shock_mode        = SHOCK_MODE,
        enable_volume_filter     = True,
        enable_volatility_filter = False,
        enable_atr_filter        = False,
        enable_market_filter     = False,
        sym_active_df            = sym_active_df,
    )


# ------------------------------------------------------------------ #
# 1パターンの WF全セグ + true OOS 実行
# ------------------------------------------------------------------ #
def run_pattern(pattern_name, cfg_pat,
                universe_raw, rsr_df, alpha_df, regime_df, all_syms,
                tech_matrices, topix_close, base_cfg):
    print(f"\n{'─'*60}")
    print(f"  パターン {pattern_name}")
    print(f"  cluster_cap={cfg_pat.risk_controls.cluster_cap:.2f}  "
          f"bear_sector_cap={cfg_pat.risk_controls.bear_sector_cap:.2f}  "
          f"bear_cluster_cap={cfg_pat.risk_controls.bear_cluster_cap:.2f}")

    seg_results = []
    wf_wins = 0

    for seg_def in WF_SEGS:
        n = seg_def["seg"]
        is_s, is_e = seg_def["is"]
        oos_s, oos_e = seg_def["oos"]

        active_is  = _build_active(universe_raw, topix_close, all_syms, rsr_df,
                                    BULL_ACTIVE_N, BEAR_ACTIVE_N, "2017-01-01", is_e, None)
        active_oos = _build_active(universe_raw, topix_close, all_syms, rsr_df,
                                    BULL_ACTIVE_N, BEAR_ACTIVE_N, "2017-01-01", oos_e, None)

        r_is  = run_seg(universe_raw, rsr_df, alpha_df, regime_df, all_syms,
                        tech_matrices, topix_close, cfg_pat, is_s, is_e, active_is)
        r_oos = run_seg(universe_raw, rsr_df, alpha_df, regime_df, all_syms,
                        tech_matrices, topix_close, cfg_pat, oos_s, oos_e, active_oos)

        is_sh  = r_is.get("sharpe", 0)
        oos_sh = r_oos.get("sharpe", 0)
        oos_dd = r_oos.get("max_dd", 0)
        win    = oos_sh > 0
        if win:
            wf_wins += 1

        seg_results.append({
            "seg": n, "oos_year": oos_s[:4],
            "is_sharpe": round(is_sh, 3), "oos_sharpe": round(oos_sh, 3),
            "oos_max_dd": round(oos_dd, 2), "win": win,
        })
        mark = "OK" if win else "NG"
        print(f"    Seg{n} OOS={oos_s[:4]}  IS={is_sh:.3f}  OOS={oos_sh:.3f}"
              f"  DD={oos_dd:+.1f}%  {mark}")
        del r_is, r_oos

    # Full IS (2018-2024)
    active_full = _build_active(universe_raw, topix_close, all_syms, rsr_df,
                                 BULL_ACTIVE_N, BEAR_ACTIVE_N, "2017-01-01", IS_FULL[1], None)
    r_is_full = run_seg(universe_raw, rsr_df, alpha_df, regime_df, all_syms,
                        tech_matrices, topix_close, cfg_pat, IS_FULL[0], IS_FULL[1], active_full)

    # True OOS 2025
    active_2025 = _build_active(universe_raw, topix_close, all_syms, rsr_df,
                                  BULL_ACTIVE_N, BEAR_ACTIVE_N, "2017-01-01", TRUE_OOS[1], None)
    r_oos_2025 = run_seg(universe_raw, rsr_df, alpha_df, regime_df, all_syms,
                         tech_matrices, topix_close, cfg_pat, TRUE_OOS[0], TRUE_OOS[1], active_2025)

    is_sh_full  = r_is_full.get("sharpe", 0)
    is_dd_full  = r_is_full.get("max_dd", 0)
    oos_sh_2025 = r_oos_2025.get("sharpe", 0)
    oos_dd_2025 = r_oos_2025.get("max_dd", 0)
    oos_cagr_2025 = r_oos_2025.get("cagr", 0)
    is_cagr_full  = r_is_full.get("cagr", 0)

    print(f"\n    Full IS (2018-2024): CAGR={is_cagr_full:+.1f}%  Sharpe={is_sh_full:.3f}"
          f"  MaxDD={is_dd_full:.1f}%")
    print(f"    True OOS 2025:       CAGR={oos_cagr_2025:+.1f}%  Sharpe={oos_sh_2025:.3f}"
          f"  MaxDD={oos_dd_2025:.1f}%")
    print(f"    WF wins: {wf_wins}/5")

    # 2022 MaxDD（Seg3 OOS = 2022）
    seg3 = next((s for s in seg_results if s["oos_year"] == "2022"), None)
    dd_2022 = seg3["oos_max_dd"] if seg3 else float("nan")
    sh_2022 = seg3["oos_sharpe"] if seg3 else float("nan")

    # Cluster DD 寄与分析（Full IS トレードから、trade_syms でセクターを引く）
    cluster_dd = _calc_cluster_dd_contribution(r_is_full, all_syms)

    return {
        "pattern": pattern_name,
        "cluster_cap": cfg_pat.risk_controls.cluster_cap,
        "bear_sector_cap": cfg_pat.risk_controls.bear_sector_cap,
        "bear_cluster_cap": cfg_pat.risk_controls.bear_cluster_cap,
        "wf_wins": wf_wins,
        "seg_results": seg_results,
        "full_is": {
            "cagr": round(is_cagr_full, 2),
            "sharpe": round(is_sh_full, 3),
            "max_dd": round(is_dd_full, 2),
        },
        "oos_2022": {"max_dd": round(dd_2022, 2), "sharpe": round(sh_2022, 3)},
        "oos_2025": {
            "cagr": round(oos_cagr_2025, 2),
            "sharpe": round(oos_sh_2025, 3),
            "max_dd": round(oos_dd_2025, 2),
        },
        "cluster_dd_contribution": cluster_dd,
    }


def _calc_cluster_dd_contribution(result: dict, trade_syms: dict) -> dict:
    """
    トレードリストからクラスター別の損失合計を計算し、DD 寄与率を推定する。
    trade_syms: {symbol: sector} でセクターを引く（trade dict に sector がないため）
    """
    from src.strategy.cluster import get_cluster, CLUSTER_MAP_DEFAULT
    trades = result.get("_trades", [])
    if not trades:
        return {}

    total_loss = 0.0
    cluster_loss: dict[str, float] = {}
    for t in trades:
        pnl = t.get("pnl", 0.0) or 0.0
        if pnl >= 0:
            continue
        sym    = t.get("symbol", "")
        cluster = get_cluster(sym, trade_syms, CLUSTER_MAP_DEFAULT)
        cluster_loss[cluster] = cluster_loss.get(cluster, 0.0) + pnl
        total_loss += pnl

    if total_loss == 0:
        return {}

    return {
        c: round(v / total_loss * 100, 1)
        for c, v in sorted(cluster_loss.items(), key=lambda x: x[1])
    }


# ------------------------------------------------------------------ #
# メイン
# ------------------------------------------------------------------ #
def main():
    base_cfg = load_strategy_config()
    base_cfg.risk_controls  # キャッシュ確認

    print("=" * 72)
    print("  WF Cap 比較検証: dynamic_cap feature flag + cluster cap + bear adaptive")
    print("=" * 72 + "\n")

    (universe_raw, rsr_df, rsr_df_full, alpha_df, regime_df, rsr_syms, all_syms,
     tech_matrices, topix_close, base_cfg) = load_data(base_cfg)

    # 3パターンの cfg を作成
    cfg_A = _make_cfg(base_cfg,
                      cluster_cap=1.0,    # cap 無効化（実質無制限）
                      bear_sector_cap=base_cfg.risk_controls.sector_cap,  # 0.25
                      bear_cluster_cap=1.0)

    cfg_B = _make_cfg(base_cfg,
                      cluster_cap=0.35,
                      bear_sector_cap=base_cfg.risk_controls.sector_cap,  # 0.25（bear も同じ）
                      bear_cluster_cap=0.35)

    cfg_C = _make_cfg(base_cfg,
                      cluster_cap=0.35,
                      bear_sector_cap=0.18,   # Bear adaptive
                      bear_cluster_cap=0.25)

    # rsr42 のみ（all_syms → rsr_syms に絞って実行）
    results_all = []

    for pat_name, cfg_pat in [("A_fixed_only", cfg_A), ("B_cluster35", cfg_B), ("C_bear_adaptive", cfg_C)]:
        result = run_pattern(
            pat_name, cfg_pat,
            universe_raw, rsr_df, alpha_df, regime_df, rsr_syms,
            tech_matrices, topix_close, base_cfg,
        )
        results_all.append(result)

    # ── 最終比較表 ──────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  最終比較")
    print("=" * 72)

    # Cluster DD 寄与（パターン A の Full IS から）
    res_A = results_all[0]
    print("\n=== Step 2: Cluster Cap 採用判定 ===")
    print("2022年 Cluster別 DD寄与（パターンA - 固定cap基準）:")
    for c, pct in res_A.get("cluster_dd_contribution", {}).items():
        print(f"  {c:<18}: {pct:+.1f}%")

    print(f"\n{'パターン':<20} {'2022 MaxDD':>12} {'2025 Sharpe':>12} {'2025 CAGR':>12} {'WF wins':>8}")
    print("-" * 70)
    for r in results_all:
        print(f"  {r['pattern']:<18} {r['oos_2022']['max_dd']:>+12.1f} "
              f"{r['oos_2025']['sharpe']:>12.3f} {r['oos_2025']['cagr']:>+12.1f}%  "
              f"{r['wf_wins']:>4}/5")

    # 採用判定（パターン A を基準に比較）
    dd_A    = results_all[0]["oos_2022"]["max_dd"]
    sh25_A  = results_all[0]["oos_2025"]["sharpe"]

    print("\n採用判定:")
    for r in results_all[1:]:
        dd_imp  = dd_A - r["oos_2022"]["max_dd"]    # 改善 = 正（DD が小さくなる）
        sh_deg  = sh25_A - r["oos_2025"]["sharpe"]  # 劣化 = 正
        dd_ok   = dd_imp >= 2.0
        sh_ok   = sh_deg <= 0.10
        verdict = "ADOPT" if dd_ok and sh_ok else "REJECT"
        print(f"  {r['pattern']}: 2022 MaxDD改善={dd_imp:+.2f}pt {'✅' if dd_ok else '❌'}  "
              f"2025 Sharpe劣化={sh_deg:+.3f} {'✅' if sh_ok else '❌'}  → {verdict}")

    # JSON 保存
    out = {
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "patterns": results_all,
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n結果保存: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
