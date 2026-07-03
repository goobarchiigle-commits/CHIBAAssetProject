"""
scripts/run_wf_abcd_compare.py

WF A/B/C/D 4パターン比較
  A: fixed cap のみ（cluster_cap=無効、gross_exposure=無効）
  B: cluster_cap=0.35
  C: cluster_cap=0.35 + bear adaptive cap（bear_sector=0.18, bear_cluster=0.25）
  D: fixed cap + gross_exposure 制御（cluster cap なし）← 2026-04-07 追加

実行:
  cd C:/ai-trading
  python src/scripts/run_wf_abcd_compare.py
"""

from __future__ import annotations
import os, sys, json, warnings, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.pop("DATA_VERSION", None)
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import src.backtest.composite_alpha_bt as _bt
from src.backtest.wf_dynamic_universe import (
    load_data, _build_active,
    BULL_ACTIVE_N, BEAR_ACTIVE_N,
    WF_SEGS, TRUE_OOS, IS_FULL, OUTPUT_DIR
)
from src.config_loader import load_strategy_config, RiskControlsConfig, StrategyConfig

SHOCK_MODE  = "composite"
OUTPUT_JSON = os.path.join(OUTPUT_DIR, f"wf_abcd_compare_{time.strftime('%Y-%m-%d')}.json")


# ------------------------------------------------------------------ #
# cfg ヘルパー
# ------------------------------------------------------------------ #
def _make_cfg(base_cfg, cluster_cap: float, bear_sector_cap: float, bear_cluster_cap: float,
              gross_exposure_enabled: bool = False,
              gross_cap_normal: float = 1.0,
              gross_cap_drawdown_5pct: float = 0.6,
              gross_cap_drawdown_8pct: float = 0.4):
    new_rc = RiskControlsConfig(
        shock_exit_mode  = base_cfg.risk_controls.shock_exit_mode,
        regime_sizing    = base_cfg.risk_controls.regime_sizing,
        bear_scale       = base_cfg.risk_controls.bear_scale,
        dynamic_cap      = False,
        symbol_cap       = base_cfg.risk_controls.symbol_cap,
        sector_cap       = base_cfg.risk_controls.sector_cap,
        cluster_cap      = cluster_cap,
        bear_sector_cap  = bear_sector_cap,
        bear_cluster_cap = bear_cluster_cap,
        gross_exposure_enabled     = gross_exposure_enabled,
        gross_cap_normal           = gross_cap_normal,
        gross_cap_drawdown_5pct    = gross_cap_drawdown_5pct,
        gross_cap_drawdown_8pct    = gross_cap_drawdown_8pct,
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
# スキップ統計集計
# ------------------------------------------------------------------ #
def _accum_skip(agg: dict, res: dict) -> None:
    ss = res.get("skip_stats", {})
    for k in ("sector_cap", "cluster_cap", "bear_adaptive", "gross_exposure"):
        agg[k] = agg.get(k, 0) + ss.get(k, 0)
    agg["trades"]          = agg.get("trades", 0) + res.get("n_trades", 0)
    agg["gross_cap_days"]  = agg.get("gross_cap_days", 0) + res.get("gross_cap_active_days", 0)


# ------------------------------------------------------------------ #
# 1パターンの WF全セグ + true OOS 実行
# ------------------------------------------------------------------ #
def run_pattern(pattern_name, cfg_pat,
                universe_raw, rsr_df, alpha_df, regime_df, all_syms,
                tech_matrices, topix_close, base_cfg):
    print(f"\n{'─'*60}")
    print(f"  パターン {pattern_name}")
    rc = cfg_pat.risk_controls
    print(f"  cluster_cap={rc.cluster_cap:.2f}  bear_sector={rc.bear_sector_cap:.2f}"
          f"  bear_cluster={rc.bear_cluster_cap:.2f}  gross_exp={rc.gross_exposure_enabled}")

    seg_results = []
    wf_wins     = 0
    skip_agg    = {}

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

        _accum_skip(skip_agg, r_is)
        _accum_skip(skip_agg, r_oos)

        oos_sh = r_oos.get("sharpe", 0)
        oos_dd = r_oos.get("max_dd", 0)
        win    = oos_sh > 0
        if win:
            wf_wins += 1
        seg_results.append({
            "seg": n, "oos_year": oos_s[:4],
            "is_sharpe": round(r_is.get("sharpe", 0), 3),
            "oos_sharpe": round(oos_sh, 3),
            "oos_max_dd": round(oos_dd, 2),
            "win": win,
        })
        mark = "OK" if win else "NG"
        print(f"    Seg{n} OOS={oos_s[:4]}  IS={r_is.get('sharpe',0):.3f}  "
              f"OOS={oos_sh:.3f}  DD={oos_dd:+.1f}%  {mark}")
        del r_is, r_oos

    # Full IS
    active_full = _build_active(universe_raw, topix_close, all_syms, rsr_df,
                                 BULL_ACTIVE_N, BEAR_ACTIVE_N, "2017-01-01", IS_FULL[1], None)
    r_is_full = run_seg(universe_raw, rsr_df, alpha_df, regime_df, all_syms,
                        tech_matrices, topix_close, cfg_pat, IS_FULL[0], IS_FULL[1], active_full)
    _accum_skip(skip_agg, r_is_full)

    # True OOS 2025
    active_2025 = _build_active(universe_raw, topix_close, all_syms, rsr_df,
                                  BULL_ACTIVE_N, BEAR_ACTIVE_N, "2017-01-01", TRUE_OOS[1], None)
    r_oos_2025 = run_seg(universe_raw, rsr_df, alpha_df, regime_df, all_syms,
                         tech_matrices, topix_close, cfg_pat, TRUE_OOS[0], TRUE_OOS[1], active_2025)
    _accum_skip(skip_agg, r_oos_2025)

    oos_sh_2025 = r_oos_2025.get("sharpe", 0)
    oos_dd_2025 = r_oos_2025.get("max_dd", 0)

    print(f"\n    Full IS (2018-2024): Sharpe={r_is_full.get('sharpe',0):.3f}"
          f"  MaxDD={r_is_full.get('max_dd',0):.1f}%")
    print(f"    True OOS 2025:       Sharpe={oos_sh_2025:.3f}  MaxDD={oos_dd_2025:.1f}%")
    print(f"    WF wins: {wf_wins}/5")

    # Seg3 2022
    seg3    = next((s for s in seg_results if s["oos_year"] == "2022"), None)
    dd_2022 = seg3["oos_max_dd"] if seg3 else float("nan")
    sh_2022 = seg3["oos_sharpe"] if seg3 else float("nan")

    oos_sharpes  = [s["oos_sharpe"] for s in seg_results]
    wf_med_sh    = float(np.median(oos_sharpes)) if oos_sharpes else float("nan")
    worst_oos_dd = min(s["oos_max_dd"] for s in seg_results) if seg_results else float("nan")

    total_skips  = sum(skip_agg.get(k, 0) for k in ("sector_cap", "cluster_cap", "gross_exposure"))
    total_trades = skip_agg.get("trades", 0)
    skip_rate    = total_skips / max(1, total_trades + total_skips) * 100

    print(f"    skip_stats={skip_agg}  skip率={skip_rate:.1f}%")

    return {
        "pattern": pattern_name,
        "cluster_cap": rc.cluster_cap,
        "bear_sector_cap": rc.bear_sector_cap,
        "bear_cluster_cap": rc.bear_cluster_cap,
        "gross_exposure_enabled": rc.gross_exposure_enabled,
        "wf_wins": wf_wins,
        "wf_med_sharpe": round(wf_med_sh, 3),
        "worst_oos_dd": round(worst_oos_dd, 2),
        "seg_results": seg_results,
        "oos_2022": {"max_dd": round(dd_2022, 2), "sharpe": round(sh_2022, 3)},
        "oos_2025": {"sharpe": round(oos_sh_2025, 3), "max_dd": round(oos_dd_2025, 2)},
        "full_is": {"sharpe": round(r_is_full.get("sharpe", 0), 3),
                    "max_dd": round(r_is_full.get("max_dd", 0), 2),
                    "n_trades": r_is_full.get("n_trades", 0)},
        "skip_stats": {k: skip_agg.get(k, 0) for k in
                       ("sector_cap", "cluster_cap", "bear_adaptive", "gross_exposure")},
        "gross_cap_active_days": skip_agg.get("gross_cap_days", 0),
        "trade_count": total_trades,
        "skip_rate": round(skip_rate, 1),
    }


# ------------------------------------------------------------------ #
# メイン
# ------------------------------------------------------------------ #
def main():
    t0 = time.time()
    base_cfg = load_strategy_config()

    print("=" * 72)
    print("  WF A/B/C/D 比較（D = fixed cap + gross_exposure 制御）")
    print("=" * 72 + "\n")

    (universe_raw, rsr_df, rsr_df_full, alpha_df, regime_df, rsr_syms, all_syms,
     tech_matrices, topix_close, base_cfg) = load_data(base_cfg)

    # ── 4パターン設定 ────────────────────────────────────────────
    cfg_A = _make_cfg(base_cfg,
                      cluster_cap=1.0,
                      bear_sector_cap=base_cfg.risk_controls.sector_cap,
                      bear_cluster_cap=1.0,
                      gross_exposure_enabled=False)

    cfg_B = _make_cfg(base_cfg,
                      cluster_cap=0.35,
                      bear_sector_cap=base_cfg.risk_controls.sector_cap,
                      bear_cluster_cap=0.35,
                      gross_exposure_enabled=False)

    cfg_C = _make_cfg(base_cfg,
                      cluster_cap=0.35,
                      bear_sector_cap=0.18,
                      bear_cluster_cap=0.25,
                      gross_exposure_enabled=False)

    cfg_D = _make_cfg(base_cfg,
                      cluster_cap=1.0,               # cluster cap なし
                      bear_sector_cap=base_cfg.risk_controls.sector_cap,
                      bear_cluster_cap=1.0,
                      gross_exposure_enabled=True,   # gross exposure 制御のみ
                      gross_cap_normal=1.0,
                      gross_cap_drawdown_5pct=0.6,
                      gross_cap_drawdown_8pct=0.4)

    results_all = []
    for pat_name, cfg_pat in [
        ("A_fixed_only",      cfg_A),
        ("B_cluster35",       cfg_B),
        ("C_bear_adaptive",   cfg_C),
        ("D_gross_exposure",  cfg_D),
    ]:
        res = run_pattern(
            pat_name, cfg_pat,
            universe_raw, rsr_df, alpha_df, regime_df, rsr_syms,
            tech_matrices, topix_close, base_cfg,
        )
        results_all.append(res)

        elapsed = time.time() - t0
        if elapsed > 280:
            print(f"\n  ⚠ タイムアウト近接（{elapsed:.0f}秒）: 残パターンをスキップ")
            break

    # ── 比較表出力 ─────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  === WF A/B/C/D 比較結果 ===")
    print("=" * 72)

    labels = [r["pattern"] for r in results_all]
    hdr_pat = " ".join(f"{l:>18}" for l in labels)
    print(f"  {'指標':<24} {hdr_pat}")
    print(f"  {'-'*24} " + " ".join(["-"*18]*len(labels)))

    def _row(name, vals):
        print(f"  {name:<24} " + " ".join(f"{v:>18}" for v in vals))

    _row("WF勝率",           [f"{r['wf_wins']}/5" for r in results_all])
    _row("WF中央値Sharpe",   [f"{r['wf_med_sharpe']:.3f}" for r in results_all])
    _row("Seg3 2022 Sharpe", [f"{r['oos_2022']['sharpe']:.3f}" for r in results_all])
    _row("Seg3 2022 MaxDD",  [f"{r['oos_2022']['max_dd']:+.1f}%" for r in results_all])
    _row("OOS 2025 Sharpe",  [f"{r['oos_2025']['sharpe']:.3f}" for r in results_all])
    _row("OOS 2025 MaxDD",   [f"{r['oos_2025']['max_dd']:+.1f}%" for r in results_all])
    _row("worst OOS DD",     [f"{r['worst_oos_dd']:+.1f}%" for r in results_all])
    _row("trade count",      [f"{r['trade_count']}" for r in results_all])
    _row("skip(gross_exp)",  [f"{r['skip_stats'].get('gross_exposure', 0)}"
                               if r['gross_exposure_enabled'] else "-"
                               for r in results_all])
    _row("gross cap 発動日数", [f"{r['gross_cap_active_days']}日"
                                 if r['gross_exposure_enabled'] else "-"
                                 for r in results_all])
    _row("skip率",           [f"{r['skip_rate']:.1f}%" for r in results_all])

    # 採用判定（Aを基準）
    print("\n  採用判定:")
    if len(results_all) > 0:
        dd_A   = results_all[0]["oos_2022"]["max_dd"]
        sh25_A = results_all[0]["oos_2025"]["sharpe"]
        for r in results_all[1:]:
            dd_imp  = dd_A - r["oos_2022"]["max_dd"]
            sh_deg  = sh25_A - r["oos_2025"]["sharpe"]
            dd_ok   = dd_imp >= 2.0
            sh_ok   = sh_deg <= 0.10
            skip_ok = r["skip_rate"] <= 15.0
            verdict = "ADOPT" if (dd_ok and sh_ok and skip_ok) else "REJECT"
            print(f"    {r['pattern']}: "
                  f"2022DD改善={dd_imp:+.2f}pt {'✅' if dd_ok else '❌'}  "
                  f"2025Sharpe劣化={sh_deg:+.3f} {'✅' if sh_ok else '❌'}  "
                  f"skip率={r['skip_rate']:.1f}% {'✅' if skip_ok else '❌'}  "
                  f"→ {verdict}")

    # 特別フォーマット出力（Step 1 最終報告用）
    d = next((r for r in results_all if r["pattern"] == "A_fixed_only"), None)
    dd_pat = next((r for r in results_all if r["pattern"] == "D_gross_exposure"), None)
    if d and dd_pat:
        print("\n")
        print("=== Step 1: Gross Exposure 制御 ===")
        print(f"| {'指標':<22} | {'A fixed':>10} | {'D +gross_exp':>12} |")
        print(f"|{'-'*24}|{'-'*12}|{'-'*14}|")
        rows = [
            ("WF勝率",            f"{d['wf_wins']}/5",                       f"{dd_pat['wf_wins']}/5"),
            ("WF中央値Sharpe",    f"{d['wf_med_sharpe']:.3f}",               f"{dd_pat['wf_med_sharpe']:.3f}"),
            ("Seg3 2022 Sharpe",  f"{d['oos_2022']['sharpe']:.3f}",          f"{dd_pat['oos_2022']['sharpe']:.3f}"),
            ("Seg3 2022 MaxDD",   f"{d['oos_2022']['max_dd']:+.1f}%",        f"{dd_pat['oos_2022']['max_dd']:+.1f}%"),
            ("OOS 2025 Sharpe",   f"{d['oos_2025']['sharpe']:.3f}",          f"{dd_pat['oos_2025']['sharpe']:.3f}"),
            ("OOS 2025 MaxDD",    f"{d['oos_2025']['max_dd']:+.1f}%",        f"{dd_pat['oos_2025']['max_dd']:+.1f}%"),
            ("worst OOS DD",      f"{d['worst_oos_dd']:+.1f}%",              f"{dd_pat['worst_oos_dd']:+.1f}%"),
            ("trade count",       f"{d['trade_count']}",                     f"{dd_pat['trade_count']}"),
            ("skip(gross_exp)",   "-",                                        f"{dd_pat['skip_stats'].get('gross_exposure',0)}"),
            ("gross cap 発動日数", "-",                                       f"{dd_pat['gross_cap_active_days']}日"),
            ("skip率",            f"{d['skip_rate']:.1f}%",                  f"{dd_pat['skip_rate']:.1f}%"),
        ]
        for name, va, vd in rows:
            print(f"| {name:<22} | {va:>10} | {vd:>12} |")

        dd_imp  = d["oos_2022"]["max_dd"] - dd_pat["oos_2022"]["max_dd"]
        sh_deg  = d["oos_2025"]["sharpe"] - dd_pat["oos_2025"]["sharpe"]
        print(f"\n採用判定:")
        print(f"  D: 2022 MaxDD改善 >= 2.0pt? {'✅' if dd_imp >= 2.0 else '❌'}"
              f"  2025 Sharpe劣化 <= 0.10? {'✅' if sh_deg <= 0.10 else '❌'}"
              f"  skip率 <= 15%? {'✅' if dd_pat['skip_rate'] <= 15.0 else '❌'}")

    # ── 保存 ──────────────────────────────────────────────────
    out = {
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": round(time.time() - t0, 1),
        "wf_abcd": results_all,
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  結果保存: {OUTPUT_JSON}")
    print(f"  総実行時間: {time.time()-t0:.0f}秒")


if __name__ == "__main__":
    main()
