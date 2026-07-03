"""
study_m1_production_update_2026-07-04.py
M1採用後 Production基準値の再測定 + Study73 CAND_B(rsr_exit=75) 再測定

背景:
  M1(addon執行価格 close→open)を正式採用（ユーザー決裁2026-07-04, roadmap§2.2）。
  「BTをLiveへ合わせるため」の採用でありCAGR改善目的ではない。基準値低下も正式値として受容する。

本スクリプトの役割:
  1. M1適用後エンジンでCURRENT(D_ATR_EQ)のIS/OOS/FULL/WF5fold/Bootstrapをfresh run。
  2. 同エンジンでCAND_B(rsr_exit=75)も同様に測定し、Study73同様のCURRENT vs CAND_B比較を再構成。
  3. 旧(close執行)基準値と併記できるよう、両方をJSONに出力。

禁止: パラメータ最適化・新規特徴量・Production変更（本スクリプトはstrategy.yaml変更なし、観測のみ）。
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import date
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from src.backtest.snapshot_archaeology_202606 import build_common_dataset
import src.backtest.composite_alpha_bt as cab

TODAY_STR = date.today().strftime("%Y-%m-%d")
OUT_FILE  = ROOT / "backtests" / f"study_m1_production_update_{TODAY_STR}.json"

CAPITAL   = 3_000_000
MIN_HOLD  = 3
N_BOOT    = 500
BOOT_SEED = 42

IS_START, IS_END     = "2018-01-01", "2024-12-31"
OOS_START, OOS_END   = "2025-01-01", "2025-12-31"
FULL_START, FULL_END = "2018-01-01", "2025-12-30"
DATA_END = "2025-12-31"

YEARS_STANDALONE = [("2018", "2018-01-01", "2018-12-31"), ("2019", "2019-01-01", "2019-12-31")]

WF_SEGS = [
    {"oos_s": "2020-01-01", "oos_e": "2020-12-31", "year": "2020"},
    {"oos_s": "2021-01-01", "oos_e": "2021-12-31", "year": "2021"},
    {"oos_s": "2022-01-01", "oos_e": "2022-12-31", "year": "2022"},
    {"oos_s": "2023-01-01", "oos_e": "2023-12-31", "year": "2023"},
    {"oos_s": "2024-01-01", "oos_e": "2024-12-31", "year": "2024"},
]

CONFIGS = {
    "CURRENT": {"exit_policy": "A", "addon_policy": "D", "rsr_exit": 70.0, "label": "D_ATR_EQ(RSR70)"},
    "CAND_B":  {"exit_policy": "A", "addon_policy": "D", "rsr_exit": 75.0, "label": "D_ATR_EQ(RSR75)"},
}


def safe_float(v, default=0.0):
    try:
        f = float(v); return f if not np.isnan(f) else default
    except (TypeError, ValueError): return default


def extract_metrics(raw: dict) -> dict:
    if raw is None: return {}
    return {
        "cagr":      round(safe_float(raw.get("cagr")), 2),
        "sharpe":    round(safe_float(raw.get("sharpe")), 3),
        "max_dd":    round(safe_float(raw.get("max_dd")), 2),
        "calmar":    round(safe_float(raw.get("calmar")), 3),
        "n_trades":  int(raw.get("n_trades", 0) or 0),
        "addon_cnt": int(raw.get("addon_count", 0) or 0),
        "avg_exp":   round(safe_float(raw.get("avg_exposure", 0)), 1),
    }


def get_active(ds, all_syms, start, end):
    from src.config_loader import load_strategy_config
    cfg = load_strategy_config()
    bc  = cfg.risk_controls.bear_universe_filter
    be  = list(bc.excluded_sectors) if bc.enabled else None
    return cab.build_dyn_rsr42_active(
        universe_raw=ds["universe_raw"], topix_close=ds["topix_close"],
        rsr_df=ds["rsr_df"], all_syms=all_syms, start=start, end=end,
        bear_exclude_sectors=be,
        sym_sector_map=dict(ds["trade_syms"]) if be else None,
    )


def run_bt(ds, sym_active_df, start, end, cfg_params) -> dict:
    return cab.run_scenario(
        scenario               = "BASELINE",
        universe_raw           = ds["universe_raw"],
        rsr_df                 = ds["rsr_df"],
        alpha_df               = None,
        regime_df              = ds["regime_df"],
        trade_syms             = ds["trade_syms"],
        rsr_syms               = ds["rsr_syms"],
        cfg                    = ds["base_cfg"],
        start                  = start, end=end, verbose=False,
        tech_matrices          = ds["tech_matrices"],
        breadth_series         = ds["breadth_series"],
        capital                = CAPITAL,
        min_hold               = MIN_HOLD,
        topix_close            = ds["topix_close"],
        market_shock_mode      = "composite",
        rsr_exit_threshold     = cfg_params["rsr_exit"],
        sym_active_df          = sym_active_df,
        enable_simple_rsr_exit = True,
        enable_atr_trailing_prod = True,
        enable_multilayer_rsr  = True,
        enable_atr_risk_sizing = False,
        enable_mtf_filter      = False,
        sizing_mode            = "existing",
        exit_policy            = cfg_params["exit_policy"],
        addon_policy           = cfg_params["addon_policy"],
        addon_size_frac        = 0.25,
        addon_atr_mult         = 1.0,
    )


def run_wf(ds, all_syms, cfg_params, label="") -> dict:
    seg_rows, cagrs, pass_cnt = [], [], 0
    for fold in WF_SEGS:
        act = get_active(ds, all_syms, fold["oos_s"], fold["oos_e"])
        m = extract_metrics(run_bt(ds, act, fold["oos_s"], fold["oos_e"], cfg_params))
        wf_pass = m["cagr"] > 0
        if wf_pass: pass_cnt += 1
        cagrs.append(m["cagr"])
        seg_rows.append({**m, "year": fold["year"], "wf_pass": wf_pass})
        print(f"    [{label}] WF {fold['year']}: CAGR={m['cagr']:+.2f}%  {'✓' if wf_pass else '✗'}")
    avg = round(float(np.mean(cagrs)), 2) if cagrs else 0.0
    std = round(float(np.std(cagrs)), 2) if cagrs else 0.0
    return {"wf_count": pass_cnt, "avg_cagr": avg, "fold_std": std, "segments": seg_rows}


def run_annual_standalone(ds, all_syms, cfg_params, label="") -> dict:
    """既知の潜在バグ(4055.T: 2020-08-11以降参加銘柄がactive_symsに漏れ、2018/2019単独runでKeyError)
    をStudy73と同一のtry/exceptで吸収する（M1と無関係・Study73でも同一エラーで確認済み）。"""
    annual = {}
    for yr, start, end in YEARS_STANDALONE:
        act = get_active(ds, all_syms, start, end)
        try:
            m = extract_metrics(run_bt(ds, act, start, end, cfg_params))
            annual[yr] = m
            print(f"    [{label}] {yr}: CAGR={m['cagr']:+.2f}%")
        except Exception as e:
            print(f"    [{label}] {yr}: ERROR {e} (既知の4055.T問題・Study73と同一)")
            annual[yr] = {"cagr": 0.0, "error": str(e)}
    return annual


def bootstrap_ci(annual_dict: dict, years_in: list, n_iter: int = N_BOOT) -> dict:
    returns = [annual_dict[yr]["cagr"] for yr in years_in if yr in annual_dict and "cagr" in annual_dict[yr]]
    if not returns:
        return {}
    n = len(returns)
    r = np.array(returns) / 100 + 1
    rng = np.random.default_rng(BOOT_SEED)
    boots = []
    for _ in range(n_iter):
        samp = rng.choice(r, size=n, replace=True)
        boots.append((float(np.prod(samp)) ** (1 / n) - 1) * 100)
    a = np.array(boots)
    return {
        "median": round(float(np.median(a)), 2), "ci_5": round(float(np.percentile(a, 5)), 2),
        "ci_95": round(float(np.percentile(a, 95)), 2), "std": round(float(np.std(a)), 2),
        "p_positive": round(float(np.mean(a > 0)), 3), "n_years": n,
    }


def main():
    print("=" * 80)
    print("  M1採用後 Production基準値 再測定 + Study73 CAND_B 再測定")
    print(f"  Date: {TODAY_STR}   Capital: Y{CAPITAL:,}")
    print("  エンジン状態: M1適用済み(addon執行=翌日寄付) / M2は変更なし(×1.5維持)")
    print("=" * 80)

    ds = build_common_dataset(DATA_END)
    all_syms = list(ds["trade_syms"].keys())
    print(f"  {len(all_syms)} symbols")

    results = {}
    for name, cfg in CONFIGS.items():
        print(f"\n[RUN] {name} ({cfg['label']})")

        print("  IS 2018-2024...")
        act_is = get_active(ds, all_syms, IS_START, IS_END)
        m_is = extract_metrics(run_bt(ds, act_is, IS_START, IS_END, cfg))
        print(f"    CAGR={m_is['cagr']:+.2f}%  Sharpe={m_is['sharpe']:.3f}  MaxDD={m_is['max_dd']:.2f}%  Calmar={m_is['calmar']:.3f}  Trades={m_is['n_trades']}  Addon={m_is['addon_cnt']}")

        print("  OOS 2025...")
        act_oos = get_active(ds, all_syms, OOS_START, OOS_END)
        m_oos = extract_metrics(run_bt(ds, act_oos, OOS_START, OOS_END, cfg))
        print(f"    CAGR={m_oos['cagr']:+.2f}%  Sharpe={m_oos['sharpe']:.3f}  MaxDD={m_oos['max_dd']:.2f}%  Calmar={m_oos['calmar']:.3f}  Trades={m_oos['n_trades']}  Addon={m_oos['addon_cnt']}")

        print("  FULL 2018-2025 (継続run)...")
        act_full = get_active(ds, all_syms, FULL_START, FULL_END)
        m_full = extract_metrics(run_bt(ds, act_full, FULL_START, FULL_END, cfg))
        print(f"    CAGR={m_full['cagr']:+.2f}%  Sharpe={m_full['sharpe']:.3f}  MaxDD={m_full['max_dd']:.2f}%  Calmar={m_full['calmar']:.3f}  Trades={m_full['n_trades']}  Addon={m_full['addon_cnt']}")

        print("  WF 5fold (2020-2024)...")
        wf = run_wf(ds, all_syms, cfg, label=name)

        print("  Annual standalone 2018-2019...")
        annual_sa = run_annual_standalone(ds, all_syms, cfg, label=name)

        annual = {}
        for seg in wf["segments"]:
            annual[seg["year"]] = seg
        annual.update(annual_sa)
        annual["2025"] = m_oos

        IS_YEARS = ["2018", "2019", "2020", "2021", "2022", "2023", "2024"]
        boot = bootstrap_ci(annual, IS_YEARS, n_iter=N_BOOT)
        print(f"  Bootstrap(N={N_BOOT}, IS年): median={boot.get('median')}%  CI=[{boot.get('ci_5')}%, {boot.get('ci_95')}%]  P(>0)={boot.get('p_positive')}")

        results[name] = {"is": m_is, "oos": m_oos, "full": m_full, "wf": wf, "annual": annual, "bootstrap": boot}

    # ── CAND_B vs CURRENT 比較 ────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("  CURRENT vs CAND_B (M1適用後エンジン)")
    print("─" * 80)
    cur, cb = results["CURRENT"], results["CAND_B"]
    d_is  = round(cb["is"]["cagr"] - cur["is"]["cagr"], 2)
    d_oos = round(cb["oos"]["cagr"] - cur["oos"]["cagr"], 2)
    d_full = round(cb["full"]["cagr"] - cur["full"]["cagr"], 2)
    d_wf  = round(cb["wf"]["avg_cagr"] - cur["wf"]["avg_cagr"], 2)
    seg22_cur = next((s for s in cur["wf"]["segments"] if s["year"] == "2022"), {})
    seg22_cb  = next((s for s in cb["wf"]["segments"] if s["year"] == "2022"), {})
    d_2022 = round(seg22_cb.get("cagr", 0) - seg22_cur.get("cagr", 0), 2)

    print(f"  IS CAGR:   {cur['is']['cagr']:+.2f}% -> {cb['is']['cagr']:+.2f}%   Δ={d_is:+.2f}pp")
    print(f"  OOS CAGR:  {cur['oos']['cagr']:+.2f}% -> {cb['oos']['cagr']:+.2f}%   Δ={d_oos:+.2f}pp")
    print(f"  FULL CAGR: {cur['full']['cagr']:+.2f}% -> {cb['full']['cagr']:+.2f}%   Δ={d_full:+.2f}pp")
    print(f"  WF avg:    {cur['wf']['avg_cagr']:+.2f}% -> {cb['wf']['avg_cagr']:+.2f}%   Δ={d_wf:+.2f}pp   WF_pass {cur['wf']['wf_count']}/5 -> {cb['wf']['wf_count']}/5")
    print(f"  2022:      {seg22_cur.get('cagr',0):+.2f}% -> {seg22_cb.get('cagr',0):+.2f}%   Δ={d_2022:+.2f}pp")
    print(f"  Bootstrap P(>0): {cur['bootstrap'].get('p_positive')} -> {cb['bootstrap'].get('p_positive')}")

    gate_pass = (cb["wf"]["wf_count"] >= 5) and (d_2022 > 0)
    print(f"\n  CAND_B採用ゲート(WF5/5 ∧ 2022改善): {'PASS' if gate_pass else 'FAIL'}")

    output = {
        "study": "M1_production_update", "date": TODAY_STR,
        "engine_state": "M1適用済み(addon=open執行)/M2変更なし(×1.5維持)",
        "results": results,
        "cand_b_comparison": {
            "delta_is": d_is, "delta_oos": d_oos, "delta_full": d_full, "delta_wf_avg": d_wf,
            "delta_2022": d_2022, "wf_pass_current": cur["wf"]["wf_count"], "wf_pass_cand_b": cb["wf"]["wf_count"],
            "gate_pass": gate_pass,
        },
    }
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[OUTPUT] {OUT_FILE}")


if __name__ == "__main__":
    main()
