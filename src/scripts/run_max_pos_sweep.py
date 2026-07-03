"""
scripts/run_max_pos_sweep.py
3ステップ改善検証スクリプト

Step 1: max_positions sweep [3, 4, 5, 6]
  - equal-weight sizing: alloc = capital / max_positions
  - gross_exposure = 1.0 固定（既存設定を維持）
  - WF5セグメント + Full IS + True OOS 2025

Step 2: Bear mode スコアリング改良 (v1 vs v2)
  - v1: 既存（rs_topix=0.50, rsr=0.30, log_vol=0.20）
  - v2: low_beta追加（rs_topix=0.45, rsr=0.25, low_beta=0.20, defensive=0.10）
  - Step 1 の最良 max_positions で実行

Step 3: Bear universe cyclical フィルター（on/off）
  - 除外: 機械/鉄鋼/銀行/保険/輸送機器/海運/化学
  - Step 1 の最良 N + Step 2 の勝者スコアで実行

実行:
  cd C:/ai-trading
  python src/scripts/run_max_pos_sweep.py
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
    WF_SEGS, TRUE_OOS, IS_FULL, OUTPUT_DIR,
)
from src.config_loader import load_strategy_config, PortfolioConfig, StrategyConfig
from src.strategy.universe import build_sym_active_df

SHOCK_MODE  = "composite"
DATA_START  = "2017-01-01"
TIMEOUT_SEC = 540  # 9分でStep1を打ち切り

# Bear cyclical 除外セクター（CSVのセクター略称）
CYCLICAL_SECTORS_BEAR = ["機械", "鉄鋼", "銀行", "保険", "輸送機器", "海運", "化学"]
DEFENSIVE_SECTORS_V2  = {"食品", "ガス", "情報通信"}

OUTPUT_JSON = os.path.join(OUTPUT_DIR, f"max_pos_sweep_{time.strftime('%Y-%m-%d')}.json")


# ------------------------------------------------------------------ #
# cfg ヘルパー
# ------------------------------------------------------------------ #
def _make_cfg_with_n(base_cfg, max_positions: int) -> StrategyConfig:
    """max_positions だけ変えた StrategyConfig を返す。"""
    new_port = PortfolioConfig(
        capital           = base_cfg.portfolio.capital,
        max_positions     = max_positions,
        min_sectors       = base_cfg.portfolio.min_sectors,
        max_single_weight = 1.0,   # equal sizing が意図した weight なので upper cap を外す
        max_dd_limit      = base_cfg.portfolio.max_dd_limit,
        vol_target        = base_cfg.portfolio.vol_target,
        use_idm           = base_cfg.portfolio.use_idm,
    )
    return StrategyConfig(
        fujiko         = base_cfg.fujiko,
        mean_reversion = base_cfg.mean_reversion,
        portfolio      = new_port,
        risk           = base_cfg.risk,
        risk_controls  = base_cfg.risk_controls,
    )


# ------------------------------------------------------------------ #
# sym_active_df ビルダー（bear_scoring_version / bear_exclude_sectors 対応）
# ------------------------------------------------------------------ #
def _build_active_ex(
    universe_raw, topix_close, all_syms, rsr_df,
    active_n, bear_n, data_start, end,
    bear_scoring_version: str = "v1",
    bear_exclude_sectors=None,
    sym_sector_map=None,
):
    """build_sym_active_df の拡張ラッパー（v2 scoring / sector filter 対応）。"""
    return build_sym_active_df(
        universe_raw         = universe_raw,
        topix_close          = topix_close,
        all_syms             = list(all_syms.keys()),
        active_n             = active_n,
        start                = data_start,
        end                  = end,
        rsr_df_full          = rsr_df,
        bear_n               = bear_n,
        bear_pool            = None,
        bear_scoring_version = bear_scoring_version,
        bear_exclude_sectors = bear_exclude_sectors,
        sym_sector_map       = sym_sector_map,
    )


# ------------------------------------------------------------------ #
# 1セグメント実行
# ------------------------------------------------------------------ #
def run_seg(
    universe_raw, rsr_df, alpha_df, regime_df, all_syms,
    tech_matrices, topix_close, cfg,
    start: str, end: str,
    sym_active_df=None,
    max_positions: int = 3,
) -> dict:
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
        sizing_mode              = "equal",
    )


# ------------------------------------------------------------------ #
# 1設定の WF全セグ + IS + OOS2025 実行
# ------------------------------------------------------------------ #
def run_pattern(
    label: str,
    max_positions: int,
    cfg,
    universe_raw, rsr_df, alpha_df, regime_df, all_syms,
    tech_matrices, topix_close,
    bear_scoring_version: str = "v1",
    bear_exclude_sectors=None,
    sym_sector_map=None,
    timeout_at: float | None = None,
) -> dict | None:
    print(f"\n{'─'*60}")
    print(f"  {label}  (N={max_positions}, bear_scoring={bear_scoring_version},"
          f" filter={'on' if bear_exclude_sectors else 'off'})")

    seg_results = []
    wf_wins = 0

    for seg_def in WF_SEGS:
        if timeout_at and time.time() > timeout_at:
            print(f"  ⚠ タイムアウト ({label}): 残セグメントをスキップ")
            return None

        n = seg_def["seg"]
        is_s, is_e = seg_def["is"]
        oos_s, oos_e = seg_def["oos"]

        active_is  = _build_active_ex(universe_raw, topix_close, all_syms, rsr_df,
                                       BULL_ACTIVE_N, BEAR_ACTIVE_N, DATA_START, is_e,
                                       bear_scoring_version, bear_exclude_sectors, sym_sector_map)
        active_oos = _build_active_ex(universe_raw, topix_close, all_syms, rsr_df,
                                       BULL_ACTIVE_N, BEAR_ACTIVE_N, DATA_START, oos_e,
                                       bear_scoring_version, bear_exclude_sectors, sym_sector_map)

        r_is  = run_seg(universe_raw, rsr_df, alpha_df, regime_df, all_syms,
                        tech_matrices, topix_close, cfg, is_s, is_e, active_is, max_positions)
        r_oos = run_seg(universe_raw, rsr_df, alpha_df, regime_df, all_syms,
                        tech_matrices, topix_close, cfg, oos_s, oos_e, active_oos, max_positions)

        oos_sh = r_oos.get("sharpe", 0)
        oos_dd = r_oos.get("max_dd", 0)
        win    = oos_sh > 0
        if win:
            wf_wins += 1

        seg_results.append({
            "seg": n, "oos_year": oos_s[:4],
            "is_sharpe":  round(r_is.get("sharpe", 0), 3),
            "oos_sharpe": round(oos_sh, 3),
            "oos_max_dd": round(oos_dd, 2),
            "win": win,
        })
        mark = "OK" if win else "NG"
        print(f"    Seg{n} OOS={oos_s[:4]}  IS={r_is.get('sharpe',0):.3f}"
              f"  OOS={oos_sh:.3f}  DD={oos_dd:+.1f}%  {mark}")
        del r_is, r_oos

    # Full IS
    active_full = _build_active_ex(universe_raw, topix_close, all_syms, rsr_df,
                                    BULL_ACTIVE_N, BEAR_ACTIVE_N, DATA_START, IS_FULL[1],
                                    bear_scoring_version, bear_exclude_sectors, sym_sector_map)
    r_full = run_seg(universe_raw, rsr_df, alpha_df, regime_df, all_syms,
                     tech_matrices, topix_close, cfg, *IS_FULL, active_full, max_positions)

    # True OOS 2025
    active_2025 = _build_active_ex(universe_raw, topix_close, all_syms, rsr_df,
                                    BULL_ACTIVE_N, BEAR_ACTIVE_N, DATA_START, TRUE_OOS[1],
                                    bear_scoring_version, bear_exclude_sectors, sym_sector_map)
    r_oos25 = run_seg(universe_raw, rsr_df, alpha_df, regime_df, all_syms,
                      tech_matrices, topix_close, cfg, *TRUE_OOS, active_2025, max_positions)

    oos_sharpes  = [s["oos_sharpe"] for s in seg_results]
    wf_med_sh    = float(np.median(oos_sharpes)) if oos_sharpes else float("nan")
    seg3         = next((s for s in seg_results if s["oos_year"] == "2022"), None)
    sh_2022      = seg3["oos_sharpe"] if seg3 else float("nan")
    dd_2022      = seg3["oos_max_dd"] if seg3 else float("nan")
    full_sh      = r_full.get("sharpe", float("nan"))
    full_dd      = r_full.get("max_dd", float("nan"))
    oos25_sh     = r_oos25.get("sharpe", float("nan"))
    oos25_dd     = r_oos25.get("max_dd", float("nan"))
    n_trades     = r_full.get("n_trades", 0)
    n_trades_yr  = r_full.get("n_trades_yr", 0.0)
    years_full   = 7.0  # IS_FULL 2018-2024

    # turnover 近似: 年間取引回数 × 1ポジション当たり比率
    turnover_yr = n_trades_yr / max_positions * 100.0  # %/年（概算）

    # avg pairwise corr（全銘柄年間リターン相関の上三角平均）
    # 注: 実際の保有銘柄間相関の近似値（安価な計算）
    try:
        close_dict = {s: universe_raw[s]["df"]["Close"] for s in all_syms if s in universe_raw}
        close_df = pd.DataFrame(close_dict).sort_index()
        close_df = close_df.loc["2018-01-01":"2024-12-31"]
        ret_df = close_df.pct_change().dropna(how="all")
        ret_df = ret_df.dropna(axis=1, how="any")
        if ret_df.shape[1] >= 2:
            corr_mat = ret_df.corr().values
            n_c = len(corr_mat)
            upper_idx = np.triu_indices(n_c, k=1)
            avg_pairwise_corr = float(np.nanmean(corr_mat[upper_idx]))
        else:
            avg_pairwise_corr = float("nan")
    except Exception:
        avg_pairwise_corr = float("nan")

    # bear_days: sym_active_df の bear 判定日数（近似: full IS のみ）
    # 詳細は省略してセグメント情報から推定しない

    print(f"  Full IS (2018-2024): Sharpe={full_sh:.3f}  MaxDD={full_dd:.1f}%")
    print(f"  True OOS 2025: Sharpe={oos25_sh:.3f}  MaxDD={oos25_dd:.1f}%")
    print(f"  WF wins: {wf_wins}/5  median={wf_med_sh:.3f}")
    print(f"  Trades(IS full): {n_trades}  ({n_trades_yr:.0f}/yr)")

    return {
        "label":             label,
        "max_positions":     max_positions,
        "bear_scoring":      bear_scoring_version,
        "bear_filter":       bool(bear_exclude_sectors),
        "wf_wins":           wf_wins,
        "wf_med_sharpe":     round(wf_med_sh, 3),
        "seg3_2022_sharpe":  round(sh_2022, 3),
        "seg3_2022_maxdd":   round(dd_2022, 2),
        "oos_2025_sharpe":   round(oos25_sh, 3),
        "oos_2025_maxdd":    round(oos25_dd, 2),
        "full_sharpe":       round(full_sh, 3),
        "full_maxdd":        round(full_dd, 2),
        "n_trades":          n_trades,
        "n_trades_yr":       round(n_trades_yr, 1),
        "turnover_yr_pct":   round(turnover_yr, 1),
        "avg_pairwise_corr": round(avg_pairwise_corr, 3),
        "seg_results":       seg_results,
    }


# ------------------------------------------------------------------ #
# 結果表示
# ------------------------------------------------------------------ #
def _print_step1_table(results: list[dict]) -> None:
    cols = [r for r in results if r is not None]
    ns   = [r["max_positions"] for r in cols]
    hdr  = " | ".join(f"N={n:1d}   " for n in ns)
    print(f"\n=== max_positions Sweep（equal-weight sizing, gross=1.0） ===\n")
    print(f"| {'指標':<24} | {hdr} |")
    print(f"|{'─'*26}|" + "".join([f"{'─'*8}|"] * len(cols)))

    def _row(name, vals):
        print(f"| {name:<24} | {' | '.join(f'{v:<6}' for v in vals)} |")

    _row("WF勝率",           [f"{r['wf_wins']}/5" for r in cols])
    _row("WF中央値Sharpe",   [f"{r['wf_med_sharpe']:.3f}" for r in cols])
    _row("Seg3 2022 Sharpe", [f"{r['seg3_2022_sharpe']:.3f}" for r in cols])
    _row("Seg3 2022 MaxDD",  [f"{r['seg3_2022_maxdd']:.1f}%" for r in cols])
    _row("OOS 2025 Sharpe",  [f"{r['oos_2025_sharpe']:.3f}" for r in cols])
    _row("OOS 2025 MaxDD",   [f"{r['oos_2025_maxdd']:.1f}%" for r in cols])
    _row("Full Sharpe",      [f"{r['full_sharpe']:.3f}" for r in cols])
    _row("trade count(IS)",  [f"{r['n_trades']}" for r in cols])
    _row("turnover (年率)",  [f"{r['turnover_yr_pct']:.0f}%" for r in cols])
    _row("avg pairwise corr",[f"{r['avg_pairwise_corr']:.3f}" for r in cols])

    # 基準: N=3
    base = cols[0]
    print(f"\n採用基準:")
    print(f"  2022 DD改善 ≥ 2.0pt vs N=3")
    print(f"  2025 Sharpe劣化 ≤ 0.15 vs N=3")
    print(f"  turnover増加 ≤ 25%")
    print(f"\n採用判定:")
    for r in cols[1:]:
        dd_imp = base["seg3_2022_maxdd"] - r["seg3_2022_maxdd"]  # 改善 = DD が浅くなる (+)
        sh_deg = base["oos_2025_sharpe"] - r["oos_2025_sharpe"]  # 劣化 = 差が + なら悪化
        to_inc = r["turnover_yr_pct"] - base["turnover_yr_pct"]
        dd_ok  = dd_imp >= 2.0
        sh_ok  = sh_deg <= 0.15
        to_ok  = to_inc <= 25.0
        all_ok = dd_ok and sh_ok and to_ok
        mark   = "✅" if all_ok else "❌"
        print(f"  N={r['max_positions']}: DD改善={dd_imp:+.1f}pt {'✅' if dd_ok else '❌'}"
              f"  Sharpe劣化={sh_deg:+.3f} {'✅' if sh_ok else '❌'}"
              f"  turnover増加={to_inc:+.0f}% {'✅' if to_ok else '❌'}"
              f"  → {mark}")


def _print_step2_table(v1: dict, v2: dict, best_n: int) -> None:
    print(f"\n=== Bear Scoring: v1 vs v2 ===\n（N={best_n}で実行）\n")
    print(f"| {'指標':<22} | {'v1 (現行)':>12} | {'v2 (low_beta追加)':>18} |")
    print(f"|{'─'*24}|{'─'*14}|{'─'*20}|")

    def _row(name, k, fmt="{:.3f}"):
        v1v = fmt.format(v1.get(k, float("nan")))
        v2v = fmt.format(v2.get(k, float("nan")))
        print(f"| {name:<22} | {v1v:>12} | {v2v:>18} |")

    _row("Seg3 2022 Sharpe", "seg3_2022_sharpe")
    _row("Seg3 2022 MaxDD",  "seg3_2022_maxdd",  "{:.1f}%")
    _row("OOS 2025 Sharpe",  "oos_2025_sharpe")
    _row("OOS 2025 MaxDD",   "oos_2025_maxdd",   "{:.1f}%")
    _row("Full Sharpe",      "full_sharpe")

    dd_imp = v1["seg3_2022_maxdd"] - v2["seg3_2022_maxdd"]
    sh_deg = v1["oos_2025_sharpe"] - v2["oos_2025_sharpe"]
    adopt  = dd_imp >= 1.0 and sh_deg <= 0.10
    print(f"\n採用判定: v2採用? {'✅' if adopt else '❌'}"
          f"（2022 DD改善={dd_imp:+.1f}pt, 2025 Sharpe変化={-sh_deg:+.3f}）")
    return adopt


def _print_step3_table(r_off: dict, r_on: dict, best_n: int, bear_ver: str) -> None:
    print(f"\n=== Bear Universe Filter: なし vs あり ===\n（N={best_n}, scoring={bear_ver}で実行）\n")
    print(f"| {'指標':<22} | {'filter=off':>12} | {'filter=on':>12} |")
    print(f"|{'─'*24}|{'─'*14}|{'─'*14}|")

    def _row(name, k, fmt="{:.3f}"):
        ov = fmt.format(r_off.get(k, float("nan")))
        nv = fmt.format(r_on.get(k, float("nan")))
        print(f"| {name:<22} | {ov:>12} | {nv:>12} |")

    _row("Seg3 2022 Sharpe", "seg3_2022_sharpe")
    _row("Seg3 2022 MaxDD",  "seg3_2022_maxdd",  "{:.1f}%")
    _row("OOS 2025 Sharpe",  "oos_2025_sharpe")
    _row("OOS 2025 MaxDD",   "oos_2025_maxdd",   "{:.1f}%")
    _row("Full Sharpe",      "full_sharpe")
    _row("trade count(IS)",  "n_trades",          "{:.0f}")

    dd_imp = r_off["seg3_2022_maxdd"] - r_on["seg3_2022_maxdd"]
    sh_deg = r_off["oos_2025_sharpe"] - r_on["oos_2025_sharpe"]
    adopt  = dd_imp >= 1.0 and sh_deg <= 0.10
    print(f"\n採用判定: filter採用? {'✅' if adopt else '❌'}"
          f"（2022 DD改善={dd_imp:+.1f}pt, 2025 Sharpe変化={-sh_deg:+.3f}）")
    return adopt


# ------------------------------------------------------------------ #
# メイン
# ------------------------------------------------------------------ #
def main():
    t0 = time.time()
    base_cfg = load_strategy_config()

    print("=" * 72)
    print("  max_positions Sweep + Bear Scoring + Bear Filter 3ステップ検証")
    print("=" * 72)

    (universe_raw, rsr_df, rsr_df_full, alpha_df, regime_df, rsr_syms, all_syms,
     tech_matrices, topix_close, base_cfg) = load_data(base_cfg)

    # sym_sector_map: symbol -> sector (for bear filter / defensive score)
    sym_sector_map = dict(all_syms)  # {symbol: sector}

    all_results = {}

    # ================================================================
    # Step 1: max_positions sweep [3, 4, 5, 6]
    # ================================================================
    print("\n" + "=" * 72)
    print("  STEP 1: max_positions Sweep [3, 4, 5, 6]")
    print("=" * 72)

    step1_results = []
    timeout_step1 = t0 + TIMEOUT_SEC

    for n in [3, 4, 5, 6]:
        if time.time() > timeout_step1:
            print(f"\n  ⚠ Step1 タイムアウト: N={n} 以降をスキップ")
            break
        cfg_n = _make_cfg_with_n(base_cfg, n)
        res = run_pattern(
            label               = f"N={n}",
            max_positions       = n,
            cfg                 = cfg_n,
            universe_raw        = universe_raw,
            rsr_df              = rsr_df,
            alpha_df            = alpha_df,
            regime_df           = regime_df,
            all_syms            = rsr_syms,
            tech_matrices       = tech_matrices,
            topix_close         = topix_close,
            bear_scoring_version= "v1",
            bear_exclude_sectors= None,
            sym_sector_map      = sym_sector_map,
            timeout_at          = timeout_step1,
        )
        if res:
            step1_results.append(res)
        all_results[f"step1_N{n}"] = res

    # Step1 結果表示
    if step1_results:
        _print_step1_table(step1_results)

    # 最良 N の決定
    base_r = next((r for r in step1_results if r["max_positions"] == 3), None)
    best_n = 3
    best_verdict = "N=3（デフォルト）"
    if base_r:
        for r in step1_results[1:]:
            dd_imp = base_r["seg3_2022_maxdd"] - r["seg3_2022_maxdd"]
            sh_deg = base_r["oos_2025_sharpe"] - r["oos_2025_sharpe"]
            to_inc = r["turnover_yr_pct"] - base_r["turnover_yr_pct"]
            if dd_imp >= 2.0 and sh_deg <= 0.15 and to_inc <= 25.0:
                # 複数条件を満たす場合は最大の DD 改善を優先
                if dd_imp > (base_r["seg3_2022_maxdd"] - (
                        next((x["seg3_2022_maxdd"] for x in step1_results
                              if x["max_positions"] == best_n), 0))):
                    best_n = r["max_positions"]
                    best_verdict = f"N={best_n}（DD改善={dd_imp:+.1f}pt）"

    print(f"\n推奨: {best_verdict}")

    elapsed1 = time.time() - t0
    print(f"\n  Step1 経過時間: {elapsed1:.0f}秒")

    # ================================================================
    # Step 2: Bear Scoring v1 vs v2
    # ================================================================
    print("\n" + "=" * 72)
    print(f"  STEP 2: Bear Scoring v1 vs v2（N={best_n}）")
    print("=" * 72)

    timeout_step2 = time.time() + 300  # 5分

    cfg_best = _make_cfg_with_n(base_cfg, best_n)

    res_v1 = run_pattern(
        label               = f"N={best_n}_bear_v1",
        max_positions       = best_n,
        cfg                 = cfg_best,
        universe_raw        = universe_raw,
        rsr_df              = rsr_df,
        alpha_df            = alpha_df,
        regime_df           = regime_df,
        all_syms            = rsr_syms,
        tech_matrices       = tech_matrices,
        topix_close         = topix_close,
        bear_scoring_version= "v1",
        bear_exclude_sectors= None,
        sym_sector_map      = sym_sector_map,
        timeout_at          = timeout_step2,
    )
    all_results["step2_v1"] = res_v1

    res_v2 = None
    if time.time() < timeout_step2:
        res_v2 = run_pattern(
            label               = f"N={best_n}_bear_v2",
            max_positions       = best_n,
            cfg                 = cfg_best,
            universe_raw        = universe_raw,
            rsr_df              = rsr_df,
            alpha_df            = alpha_df,
            regime_df           = regime_df,
            all_syms            = rsr_syms,
            tech_matrices       = tech_matrices,
            topix_close         = topix_close,
            bear_scoring_version= "v2",
            bear_exclude_sectors= None,
            sym_sector_map      = sym_sector_map,
            timeout_at          = timeout_step2,
        )
    all_results["step2_v2"] = res_v2

    best_bear_ver = "v1"
    step2_adopt   = False
    if res_v1 and res_v2:
        step2_adopt = _print_step2_table(res_v1, res_v2, best_n)
        if step2_adopt:
            best_bear_ver = "v2"
    else:
        print("  ⚠ Step2 がタイムアウトまたは失敗 → v1 を維持")

    # ================================================================
    # Step 3: Bear Universe Filter（on/off）
    # ================================================================
    print("\n" + "=" * 72)
    print(f"  STEP 3: Bear Universe Filter（N={best_n}, scoring={best_bear_ver}）")
    print("=" * 72)

    timeout_step3 = time.time() + 300  # 5分

    # filter=off は Step2 の v1/v2 結果を再利用
    res_filter_off = res_v1 if best_bear_ver == "v1" else res_v2
    all_results["step3_filter_off"] = res_filter_off

    res_filter_on = None
    if time.time() < timeout_step3:
        res_filter_on = run_pattern(
            label               = f"N={best_n}_{best_bear_ver}_filter_on",
            max_positions       = best_n,
            cfg                 = cfg_best,
            universe_raw        = universe_raw,
            rsr_df              = rsr_df,
            alpha_df            = alpha_df,
            regime_df           = regime_df,
            all_syms            = rsr_syms,
            tech_matrices       = tech_matrices,
            topix_close         = topix_close,
            bear_scoring_version= best_bear_ver,
            bear_exclude_sectors= CYCLICAL_SECTORS_BEAR,
            sym_sector_map      = sym_sector_map,
            timeout_at          = timeout_step3,
        )
    all_results["step3_filter_on"] = res_filter_on

    step3_adopt = False
    if res_filter_off and res_filter_on:
        step3_adopt = _print_step3_table(res_filter_off, res_filter_on, best_n, best_bear_ver)
    else:
        print("  ⚠ Step3 がタイムアウトまたは失敗 → filter off を維持")

    # ================================================================
    # 最終推奨
    # ================================================================
    print("\n" + "=" * 72)
    print("  === 総合推奨（3ステップ組み合わせ） ===")
    print("=" * 72)
    final_bear_ver    = "v2" if step2_adopt else "v1"
    final_filter      = "on"  if step3_adopt else "off"
    final_n           = best_n

    # 期待される改善の推定
    base3 = next((r for r in step1_results if r and r["max_positions"] == 3), None)
    final_r = res_filter_on if step3_adopt else res_filter_off

    dd_expected  = float("nan")
    sh_expected  = float("nan")
    if base3 and final_r:
        dd_expected = base3["seg3_2022_maxdd"] - final_r["seg3_2022_maxdd"]
        sh_expected = final_r["oos_2025_sharpe"] - base3["oos_2025_sharpe"]

    print(f"\n推奨設定:")
    print(f"  max_positions:          {final_n}")
    print(f"  bear_scoring:           {final_bear_ver}")
    print(f"  bear_universe_filter:   {final_filter}")
    print(f"  期待される 2022 MaxDD 改善: {dd_expected:+.1f}pt（vs 現行 N=3 v1 filter=off）")
    print(f"  期待される 2025 Sharpe 変化: {sh_expected:+.3f}")

    # ================================================================
    # 保存
    # ================================================================
    save_data = {}
    for k, v in all_results.items():
        if v is None:
            save_data[k] = None
        else:
            save_data[k] = {x: y for x, y in v.items() if x != "equity_curve"}

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"\n結果保存: {OUTPUT_JSON}")
    print(f"総経過時間: {time.time() - t0:.0f}秒")
    print("=" * 72)


if __name__ == "__main__":
    main()
