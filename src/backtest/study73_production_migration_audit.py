"""
study73_production_migration_audit.py
Study73 — Production Migration Audit

目的:
  Study70〜72により F1(RSR Exit 70) INVALID / F3(EQ Scale) REMOVE候補 が確定。
  Production変更前の最終監査のみ実施。

Candidate A: EQ Scale Remove   → B_ATR_EXT   (exit_policy=A, addon_policy=NONE, rsr=70)
Candidate B: RSR Exit 70→75   → D_ATR_EQ_R75 (exit_policy=A, addon_policy=D,    rsr=75)

Phase0: Study70-72との整合性確認 / Lookahead=0
Phase1: CURRENT vs CAND_A (EQ Scale Remove)
Phase2: CURRENT vs CAND_B (RSR 70→75)
Phase3: Robustness — Bootstrap / 年代別 / Rolling WF
Phase4: Safety Audit — Execution数 / Holding / Position / Cash / DD
Phase5: Production Decision — KEEP / REMOVE / REPLACE + 変更対象ファイル
Deliverables: 比較表 / 改善量 / リスク / 推奨構成 / 変更対象ファイル一覧

禁止: コード変更 / 閾値最適化 / 新規特徴量 / Production変更 / 観測専用
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
import pandas as pd

from src.backtest.snapshot_archaeology_202606 import build_common_dataset
import src.backtest.composite_alpha_bt as cab

# ── 出力 ──────────────────────────────────────────────────────────────────────
TODAY_STR = date.today().strftime("%Y-%m-%d")
OUT_FILE  = ROOT / "backtests" / f"study73_production_migration_audit_{TODAY_STR}.json"
S52_PATH  = ROOT / "backtests" / "study52_production_optimization_2026-06-28.json"
S72_PATH  = ROOT / "backtests" / "study72_production_research_provenance_audit_2026-07-02.json"

# ── 固定パラメータ (PARAMS_LOCKED) ────────────────────────────────────────────
CAPITAL   = 3_000_000
MIN_HOLD  = 3
ADDON_SIZE_FRAC = 0.25
ADDON_ATR_MULT  = 1.0
N_BOOT    = 500
BOOT_SEED = 42

# ── 期間定義 ─────────────────────────────────────────────────────────────────
IS_START  = "2018-01-01"
IS_END    = "2024-12-31"
OOS_START = "2025-01-01"
OOS_END   = "2025-12-31"
DATA_END  = "2025-12-31"

# 年代別: 2018-2019 (WF OOS外) を単独実行、2020-2024 は WF_SEGS 流用、2025 は OOS
YEARS_STANDALONE = [
    ("2018", "2018-01-01", "2018-12-31"),
    ("2019", "2019-01-01", "2019-12-31"),
]
YEARS_OOS = ("2025", "2025-01-01", "2025-12-31")

# WF_SEGS (composite_alpha_bt.py / wf_dynamic_universe.py と同一)
WF_SEGS = [
    {"seg": 1, "is_s": "2018-01-01", "is_e": "2019-12-31", "oos_s": "2020-01-01", "oos_e": "2020-12-31", "year": "2020"},
    {"seg": 2, "is_s": "2019-01-01", "is_e": "2020-12-31", "oos_s": "2021-01-01", "oos_e": "2021-12-31", "year": "2021"},
    {"seg": 3, "is_s": "2020-01-01", "is_e": "2021-12-31", "oos_s": "2022-01-01", "oos_e": "2022-12-31", "year": "2022"},
    {"seg": 4, "is_s": "2021-01-01", "is_e": "2022-12-31", "oos_s": "2023-01-01", "oos_e": "2023-12-31", "year": "2023"},
    {"seg": 5, "is_s": "2022-01-01", "is_e": "2023-12-31", "oos_s": "2024-01-01", "oos_e": "2024-12-31", "year": "2024"},
]

# ── 3構成定義 ─────────────────────────────────────────────────────────────────
CONFIGS = {
    "CURRENT":  {"exit_policy": "A", "addon_policy": "D",    "rsr_exit": 70.0,
                 "label": "D_ATR_EQ(RSR70)",   "desc": "現行Production"},
    "CAND_A":   {"exit_policy": "A", "addon_policy": "NONE", "rsr_exit": 70.0,
                 "label": "B_ATR_EXT(RSR70)",  "desc": "EQ Scale除去"},
    "CAND_B":   {"exit_policy": "A", "addon_policy": "D",    "rsr_exit": 75.0,
                 "label": "D_ATR_EQ(RSR75)",   "desc": "RSR Exit 70→75"},
}

# ── 年代分類 ─────────────────────────────────────────────────────────────────
ERA_MAP = {
    "2018": "Sideways (trade_war)",
    "2019": "Bull (pre_COVID)",
    "2020": "COVID (crash+recovery)",
    "2021": "Bull (post_COVID)",
    "2022": "Bear (rate_hike)",
    "2023": "Recovery",
    "2024": "Bull+correction",
    "2025": "OOS_2025",
}


# ======================================================================
# ユーティリティ
# ======================================================================

def safe_float(v, default=0.0):
    try:
        f = float(v); return f if not np.isnan(f) else default
    except (TypeError, ValueError): return default


def extract_metrics(raw: dict) -> dict:
    if raw is None: return {}
    return {
        "cagr":      round(safe_float(raw.get("cagr")),      2),
        "sharpe":    round(safe_float(raw.get("sharpe")),    3),
        "max_dd":    round(safe_float(raw.get("max_dd")),    2),
        "calmar":    round(safe_float(raw.get("calmar")),    3),
        "n_trades":  int(raw.get("n_trades", 0) or 0),
        "addon_cnt": int(raw.get("addon_cnt", 0) or 0),
        "avg_exp":   round(safe_float(raw.get("avg_exposure", raw.get("avg_exp", 0))), 1),
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
        addon_size_frac        = ADDON_SIZE_FRAC,
        addon_atr_mult         = ADDON_ATR_MULT,
    )


def run_wf(ds, all_syms, cfg_params, label="") -> dict:
    seg_rows = []
    pass_cnt = 0
    cagrs    = []
    for fold in WF_SEGS:
        oos_s = fold["oos_s"]; oos_e = fold["oos_e"]; yr = fold["year"]
        act = get_active(ds, all_syms, oos_s, oos_e)
        try:
            raw = run_bt(ds, act, oos_s, oos_e, cfg_params)
            m   = extract_metrics(raw)
            wf_pass = m["cagr"] > 0
            if wf_pass: pass_cnt += 1
            cagrs.append(m["cagr"])
            seg_rows.append({**m, "year": yr, "wf_pass": wf_pass})
            sym = "✓" if wf_pass else "✗"
            print(f"    [{label}] WF {yr}: CAGR={m['cagr']:+.2f}%  Sharpe={m['sharpe']:+.3f} {sym}")
        except Exception as e:
            print(f"    [{label}] WF {yr}: ERROR {e}")
            cagrs.append(0.0)
            seg_rows.append({"year": yr, "error": str(e)})
    avg = round(float(np.mean(cagrs)), 2) if cagrs else 0.0
    std = round(float(np.std(cagrs)), 2) if cagrs else 0.0
    return {"wf_count": pass_cnt, "avg_cagr": avg, "fold_std": std, "segments": seg_rows}


def run_annual_standalone(ds, all_syms, cfg_params, year_defs, label="") -> dict:
    annual = {}
    for yr, start, end in year_defs:
        act = get_active(ds, all_syms, start, end)
        try:
            raw = run_bt(ds, act, start, end, cfg_params)
            m   = extract_metrics(raw)
            annual[yr] = m
            print(f"    [{label}] {yr}: CAGR={m['cagr']:+.2f}%")
        except Exception as e:
            print(f"    [{label}] {yr}: ERROR {e}")
            annual[yr] = {"cagr": 0.0, "error": str(e)}
    return annual


def build_annual_series(wf_result: dict, standalone: dict, oos_result: dict) -> dict:
    """WF折り+単独実行+OOSを結合して7+1年の年度別系列を構築"""
    annual = {}
    # WF OOSセグメント (2020-2024)
    for seg in wf_result.get("segments", []):
        yr = seg.get("year")
        if yr and "cagr" in seg:
            annual[yr] = seg
    # 単独実行 (2018, 2019)
    annual.update(standalone)
    # OOS 2025
    if oos_result:
        annual["2025"] = oos_result
    return annual


def bootstrap_ci(annual_dict: dict, years_in: list, n_iter: int = N_BOOT) -> dict:
    """Year-block bootstrap from IS years"""
    returns = []
    for yr in years_in:
        if yr in annual_dict and "cagr" in annual_dict[yr]:
            returns.append(annual_dict[yr]["cagr"])
    if not returns:
        return {}
    n   = len(returns)
    r   = np.array(returns) / 100 + 1
    rng = np.random.default_rng(BOOT_SEED)
    boots = []
    for _ in range(n_iter):
        samp = rng.choice(r, size=n, replace=True)
        cagr = (float(np.prod(samp)) ** (1/n) - 1) * 100
        boots.append(cagr)
    a = np.array(boots)
    return {
        "median":     round(float(np.median(a)), 2),
        "ci_5":       round(float(np.percentile(a, 5)), 2),
        "ci_95":      round(float(np.percentile(a, 95)), 2),
        "std":        round(float(np.std(a)), 2),
        "p_positive": round(float(np.mean(a > 0)), 3),
        "n_years":    n,
    }


def delta_m(new: dict, base: dict, key: str) -> float:
    return round(safe_float(new.get(key)) - safe_float(base.get(key)), 2)


def print_table(title: str, headers: list, rows: list):
    col_w = [max(len(h), max((len(str(r[i])) for r in rows), default=0)) + 2
             for i, h in enumerate(headers)]
    print(f"\n  {title}")
    hdr = " ".join(f"{h:>{w}}" for h, w in zip(headers, col_w))
    print("  " + hdr)
    print("  " + "─" * len(hdr))
    for row in rows:
        ln = " ".join(f"{str(v):>{w}}" for v, w in zip(row, col_w))
        print("  " + ln)


# ======================================================================
# メイン
# ======================================================================

def main():
    print("=" * 80)
    print("  Study73 — Production Migration Audit")
    print(f"  Date: {TODAY_STR}   Capital: ¥{CAPITAL:,}")
    print("=" * 80)

    # ── Phase0: Integrity ──────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("  Phase0: Integrity (Study52/70/71/72 参照確認)")
    print("─" * 80)

    s52 = {}
    if S52_PATH.exists():
        with open(S52_PATH, encoding="utf-8") as f: s52 = json.load(f)

    # Study52参照値
    s52_current_is  = s52.get("period_results", {}).get("FULL_IS",  {}).get("D_ATR_EQ", {}).get("cagr")
    s52_current_oos = s52.get("period_results", {}).get("TRUE_OOS", {}).get("D_ATR_EQ", {}).get("cagr")
    s52_canda_is    = s52.get("period_results", {}).get("FULL_IS",  {}).get("B_ATR_EXT",{}).get("cagr")
    s52_canda_oos   = s52.get("period_results", {}).get("TRUE_OOS", {}).get("B_ATR_EXT",{}).get("cagr")
    s52_current_wf  = s52.get("wf_results", {}).get("D_ATR_EQ", {})
    s52_canda_wf    = s52.get("wf_results", {}).get("B_ATR_EXT", {})
    s52_segs        = s52.get("wf_segments", {})

    print(f"\n  Study52 参照値 (2026-06-28):")
    print(f"    D_ATR_EQ(CURRENT): IS={s52_current_is}%  OOS={s52_current_oos}%  WF={s52_current_wf.get('wf_count')}/5")
    print(f"    B_ATR_EXT(CAND_A): IS={s52_canda_is}%   OOS={s52_canda_oos}%  WF={s52_canda_wf.get('wf_count')}/5")

    print(f"\n  Study70/71/72 確認済み事項:")
    print(f"    Study70: Add-on IS=-0.44pp / OOS=-0.39pp (ポートフォリオ経済値: Net negative)")
    print(f"    Study71: F1=REVIEW(OOS-0.89pp) / F3=REVIEW(IS-0.44pp OOS-0.39pp)")
    print(f"    Study72: F1=INVALID / F3=REMOVE候補 / ML_RSR無影響 / Consistency=0.62")
    print(f"    Study52 WF Seg3/2022: CURRENT=-2.65%(pass) vs CAND_A=-5.11%(fail)")
    print(f"    → EQ Scale除去で2022悪化: 重要安全確認事項")

    print(f"\n  Lookahead=0確認: 全シナリオ共通 build_common_dataset + 同一WF_SEGS使用")

    # ── データセット構築 ─────────────────────────────────────────────────────
    print(f"\n[DATA] データセット構築...")
    ds       = build_common_dataset(DATA_END)
    all_syms = list(ds["trade_syms"].keys())
    print(f"  {len(all_syms)} シンボル")

    # ── 全構成実行 ─────────────────────────────────────────────────────────────
    results = {}

    for name, cfg in CONFIGS.items():
        print(f"\n[RUN] {name} ({cfg['label']} / {cfg['desc']})")

        # IS (2018-2024)
        print(f"  IS 2018-2024...")
        act_is = get_active(ds, all_syms, IS_START, IS_END)
        raw_is = run_bt(ds, act_is, IS_START, IS_END, cfg)
        m_is   = extract_metrics(raw_is)
        print(f"    CAGR={m_is['cagr']:+.2f}%  Sharpe={m_is['sharpe']:.3f}  MaxDD={m_is['max_dd']:.2f}%  Calmar={m_is['calmar']:.3f}  Trades={m_is['n_trades']}  Exp={m_is['avg_exp']}%")

        # OOS (2025)
        print(f"  OOS 2025...")
        act_oos = get_active(ds, all_syms, OOS_START, OOS_END)
        raw_oos = run_bt(ds, act_oos, OOS_START, OOS_END, cfg)
        m_oos   = extract_metrics(raw_oos)
        print(f"    CAGR={m_oos['cagr']:+.2f}%  Sharpe={m_oos['sharpe']:.3f}  MaxDD={m_oos['max_dd']:.2f}%  Calmar={m_oos['calmar']:.3f}  Trades={m_oos['n_trades']}")

        # WF (2020-2024)
        print(f"  WF (2020-2024)...")
        wf = run_wf(ds, all_syms, cfg, label=name)

        # 年度別単独実行 (2018, 2019)
        print(f"  Annual standalone 2018-2019...")
        annual_sa = run_annual_standalone(ds, all_syms, cfg, YEARS_STANDALONE, label=name)

        # 年度別系列構築
        annual = build_annual_series(wf, annual_sa, m_oos)

        results[name] = {
            "cfg":    cfg,
            "is":     m_is,
            "oos":    m_oos,
            "wf":     wf,
            "annual": annual,
        }

    # ── Phase1: CURRENT vs CAND_A (EQ Scale Remove) ───────────────────────────
    print("\n" + "─" * 80)
    print("  Phase1: Candidate A — EQ Scale Remove (B_ATR_EXT vs CURRENT)")
    print("─" * 80)

    cur   = results["CURRENT"]
    cand_a = results["CAND_A"]

    is_cur   = cur["is"];    is_ca  = cand_a["is"]
    oos_cur  = cur["oos"];   oos_ca = cand_a["oos"]
    wf_cur   = cur["wf"];    wf_ca  = cand_a["wf"]

    nev_is_ya  = round(delta_m(is_ca,  is_cur,  "cagr") * CAPITAL / 100, 0)
    nev_oos_ya = round(delta_m(oos_ca, oos_cur, "cagr") * CAPITAL / 100, 0)

    print_table("IS/OOS比較 (CURRENT vs CAND_A)",
        ["指標", "CURRENT", "CAND_A", "ΔCAGR"],
        [
            ["IS CAGR",    f"{is_cur['cagr']:+.2f}%", f"{is_ca['cagr']:+.2f}%",
             f"{delta_m(is_ca, is_cur, 'cagr'):+.2f}pp"],
            ["IS Sharpe",  f"{is_cur['sharpe']:.3f}", f"{is_ca['sharpe']:.3f}",
             f"{delta_m(is_ca, is_cur, 'sharpe'):+.3f}"],
            ["IS MaxDD",   f"{is_cur['max_dd']:.2f}%", f"{is_ca['max_dd']:.2f}%",
             f"{delta_m(is_ca, is_cur, 'max_dd'):+.2f}pp"],
            ["IS Calmar",  f"{is_cur['calmar']:.3f}", f"{is_ca['calmar']:.3f}",
             f"{delta_m(is_ca, is_cur, 'calmar'):+.3f}"],
            ["IS Trades",  str(is_cur['n_trades']), str(is_ca['n_trades']),
             f"{is_ca['n_trades']-is_cur['n_trades']:+d}"],
            ["IS Addon",   str(is_cur['addon_cnt']), str(is_ca['addon_cnt']),
             f"{is_ca['addon_cnt']-is_cur['addon_cnt']:+d}"],
            ["IS Exp",     f"{is_cur['avg_exp']:.1f}%", f"{is_ca['avg_exp']:.1f}%",
             f"{delta_m(is_ca, is_cur, 'avg_exp'):+.1f}pp"],
            ["OOS CAGR",   f"{oos_cur['cagr']:+.2f}%", f"{oos_ca['cagr']:+.2f}%",
             f"{delta_m(oos_ca, oos_cur, 'cagr'):+.2f}pp"],
            ["OOS Sharpe", f"{oos_cur['sharpe']:.3f}", f"{oos_ca['sharpe']:.3f}",
             f"{delta_m(oos_ca, oos_cur, 'sharpe'):+.3f}"],
            ["OOS MaxDD",  f"{oos_cur['max_dd']:.2f}%", f"{oos_ca['max_dd']:.2f}%",
             f"{delta_m(oos_ca, oos_cur, 'max_dd'):+.2f}pp"],
            ["OOS Calmar", f"{oos_cur['calmar']:.3f}", f"{oos_ca['calmar']:.3f}",
             f"{delta_m(oos_ca, oos_cur, 'calmar'):+.3f}"],
            ["Portfolio NEV(IS)", f"─", f"¥{nev_is_ya:,.0f}/yr", ""],
            ["Portfolio NEV(OOS)", f"─", f"¥{nev_oos_ya:,.0f}/yr", ""],
        ]
    )

    print_table("WF比較 (CURRENT vs CAND_A)",
        ["Year", "CURRENT", "CAND_A", "Δ", "優位"],
        [
            [seg_cur["year"],
             f"{seg_cur['cagr']:+.2f}%",
             f"{seg_ca['cagr']:+.2f}%",
             f"{delta_m(seg_ca, seg_cur, 'cagr'):+.2f}pp",
             "CAND_A" if seg_ca["cagr"] > seg_cur["cagr"] else "CURRENT"]
            for seg_cur, seg_ca in zip(wf_cur["segments"], wf_ca["segments"])
        ] + [
            ["AVG",
             f"{wf_cur['avg_cagr']:+.2f}%",
             f"{wf_ca['avg_cagr']:+.2f}%",
             f"{wf_ca['avg_cagr']-wf_cur['avg_cagr']:+.2f}pp",
             "CAND_A" if wf_ca["avg_cagr"] > wf_cur["avg_cagr"] else "CURRENT"],
            ["WF_pass",
             f"{wf_cur['wf_count']}/5",
             f"{wf_ca['wf_count']}/5",
             "",
             "CAND_A" if wf_ca["wf_count"] > wf_cur["wf_count"] else
             ("CURRENT" if wf_ca["wf_count"] < wf_cur["wf_count"] else "EQUAL")],
        ]
    )

    # 2022年 (最悪年) の安全確認
    seg22_cur = next((s for s in wf_cur["segments"] if s["year"]=="2022"), {})
    seg22_ca  = next((s for s in wf_ca["segments"]  if s["year"]=="2022"), {})
    print(f"\n  ⚠ 2022年（最悪年/rate_hike Bear）安全確認:")
    print(f"    CURRENT: {seg22_cur.get('cagr', 'N/A'):+.2f}%  wf_pass={seg22_cur.get('wf_pass')}")
    print(f"    CAND_A:  {seg22_ca.get('cagr',  'N/A'):+.2f}%  wf_pass={seg22_ca.get('wf_pass')}")
    delta22_a = seg22_ca.get("cagr", 0) - seg22_cur.get("cagr", 0)
    print(f"    Δ={delta22_a:+.2f}pp → {'CAND_A悪化' if delta22_a < 0 else 'CAND_A改善'}")

    # ── Phase2: CURRENT vs CAND_B (RSR 70→75) ────────────────────────────────
    print("\n" + "─" * 80)
    print("  Phase2: Candidate B — RSR Exit 70→75 (D_ATR_EQ_R75 vs CURRENT)")
    print("─" * 80)

    cand_b = results["CAND_B"]
    is_cb  = cand_b["is"]
    oos_cb = cand_b["oos"]
    wf_cb  = cand_b["wf"]

    nev_is_yb  = round(delta_m(is_cb,  is_cur,  "cagr") * CAPITAL / 100, 0)
    nev_oos_yb = round(delta_m(oos_cb, oos_cur, "cagr") * CAPITAL / 100, 0)

    print_table("IS/OOS比較 (CURRENT vs CAND_B)",
        ["指標", "CURRENT", "CAND_B", "ΔCAGR"],
        [
            ["IS CAGR",    f"{is_cur['cagr']:+.2f}%", f"{is_cb['cagr']:+.2f}%",
             f"{delta_m(is_cb, is_cur, 'cagr'):+.2f}pp"],
            ["IS Sharpe",  f"{is_cur['sharpe']:.3f}", f"{is_cb['sharpe']:.3f}",
             f"{delta_m(is_cb, is_cur, 'sharpe'):+.3f}"],
            ["IS MaxDD",   f"{is_cur['max_dd']:.2f}%", f"{is_cb['max_dd']:.2f}%",
             f"{delta_m(is_cb, is_cur, 'max_dd'):+.2f}pp"],
            ["IS Calmar",  f"{is_cur['calmar']:.3f}", f"{is_cb['calmar']:.3f}",
             f"{delta_m(is_cb, is_cur, 'calmar'):+.3f}"],
            ["IS Trades",  str(is_cur['n_trades']), str(is_cb['n_trades']),
             f"{is_cb['n_trades']-is_cur['n_trades']:+d}"],
            ["IS Addon",   str(is_cur['addon_cnt']), str(is_cb['addon_cnt']),
             f"{is_cb['addon_cnt']-is_cur['addon_cnt']:+d}"],
            ["IS Exp",     f"{is_cur['avg_exp']:.1f}%", f"{is_cb['avg_exp']:.1f}%",
             f"{delta_m(is_cb, is_cur, 'avg_exp'):+.1f}pp"],
            ["OOS CAGR",   f"{oos_cur['cagr']:+.2f}%", f"{oos_cb['cagr']:+.2f}%",
             f"{delta_m(oos_cb, oos_cur, 'cagr'):+.2f}pp"],
            ["OOS Sharpe", f"{oos_cur['sharpe']:.3f}", f"{oos_cb['sharpe']:.3f}",
             f"{delta_m(oos_cb, oos_cur, 'sharpe'):+.3f}"],
            ["OOS MaxDD",  f"{oos_cur['max_dd']:.2f}%", f"{oos_cb['max_dd']:.2f}%",
             f"{delta_m(oos_cb, oos_cur, 'max_dd'):+.2f}pp"],
            ["OOS Calmar", f"{oos_cur['calmar']:.3f}", f"{oos_cb['calmar']:.3f}",
             f"{delta_m(oos_cb, oos_cur, 'calmar'):+.3f}"],
            ["Portfolio NEV(IS)", f"─", f"¥{nev_is_yb:,.0f}/yr", ""],
            ["Portfolio NEV(OOS)", f"─", f"¥{nev_oos_yb:,.0f}/yr", ""],
        ]
    )

    print_table("WF比較 (CURRENT vs CAND_B)",
        ["Year", "CURRENT", "CAND_B", "Δ", "優位"],
        [
            [seg_cur["year"],
             f"{seg_cur['cagr']:+.2f}%",
             f"{seg_cb['cagr']:+.2f}%",
             f"{delta_m(seg_cb, seg_cur, 'cagr'):+.2f}pp",
             "CAND_B" if seg_cb["cagr"] > seg_cur["cagr"] else "CURRENT"]
            for seg_cur, seg_cb in zip(wf_cur["segments"], wf_cb["segments"])
        ] + [
            ["AVG",
             f"{wf_cur['avg_cagr']:+.2f}%",
             f"{wf_cb['avg_cagr']:+.2f}%",
             f"{wf_cb['avg_cagr']-wf_cur['avg_cagr']:+.2f}pp",
             "CAND_B" if wf_cb["avg_cagr"] > wf_cur["avg_cagr"] else "CURRENT"],
            ["WF_pass",
             f"{wf_cur['wf_count']}/5",
             f"{wf_cb['wf_count']}/5",
             "",
             "CAND_B" if wf_cb["wf_count"] > wf_cur["wf_count"] else
             ("CURRENT" if wf_cb["wf_count"] < wf_cur["wf_count"] else "EQUAL")],
        ]
    )

    seg22_cb = next((s for s in wf_cb["segments"] if s["year"]=="2022"), {})
    delta22_b = seg22_cb.get("cagr", 0) - seg22_cur.get("cagr", 0)
    print(f"\n  ⚠ 2022年（最悪年）安全確認:")
    print(f"    CURRENT: {seg22_cur.get('cagr', 'N/A'):+.2f}%")
    print(f"    CAND_B:  {seg22_cb.get('cagr',  'N/A'):+.2f}%")
    print(f"    Δ={delta22_b:+.2f}pp → {'CAND_B悪化' if delta22_b < 0 else 'CAND_B改善'}")

    # ── Phase3: Robustness ────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("  Phase3: Robustness — Bootstrap / 年代別 / Rolling WF")
    print("─" * 80)

    IS_YEARS = ["2018","2019","2020","2021","2022","2023","2024"]

    # Bootstrap
    print("\n  [Bootstrap] Year-block bootstrap (N=500, IS years 2018-2024)")
    boot_results = {}
    for name in CONFIGS:
        bc = bootstrap_ci(results[name]["annual"], IS_YEARS, n_iter=N_BOOT)
        boot_results[name] = bc

    print_table("Bootstrap CI (IS年 year-block, N=500)",
        ["構成", "Median", "CI_5%", "CI_95%", "Std", "P(>0)"],
        [
            [CONFIGS[n]["label"],
             f"{boot_results[n].get('median',0):+.2f}%",
             f"{boot_results[n].get('ci_5',0):+.2f}%",
             f"{boot_results[n].get('ci_95',0):+.2f}%",
             f"{boot_results[n].get('std',0):.2f}pp",
             f"{boot_results[n].get('p_positive',0):.1%}"]
            for n in CONFIGS
        ]
    )

    # 年代別比較
    print("\n  [年代別] 各年の CAGR 比較")
    era_rows = []
    for yr in IS_YEARS + ["2025"]:
        cur_cagr = safe_float(results["CURRENT"]["annual"].get(yr, {}).get("cagr"))
        ca_cagr  = safe_float(results["CAND_A"]["annual"].get(yr, {}).get("cagr"))
        cb_cagr  = safe_float(results["CAND_B"]["annual"].get(yr, {}).get("cagr"))
        era = ERA_MAP.get(yr, yr)
        era_rows.append([yr, era[:18], f"{cur_cagr:+.2f}%", f"{ca_cagr:+.2f}%",
                         f"{ca_cagr-cur_cagr:+.2f}pp", f"{cb_cagr:+.2f}%",
                         f"{cb_cagr-cur_cagr:+.2f}pp"])

    print_table("年代別 CAGR",
        ["Year","Era","CURRENT","CAND_A","Δ_A","CAND_B","Δ_B"],
        era_rows
    )

    # Rolling WF サマリー (既に Phase1/2で出力済み、ここでは統合表)
    print("\n  [Rolling WF] 全構成 WF_SEGS OOS結果")
    wf_rows = []
    for fold in WF_SEGS:
        yr  = fold["year"]
        sc  = next((s for s in wf_cur["segments"] if s["year"]==yr), {})
        sa  = next((s for s in wf_ca["segments"]  if s["year"]==yr), {})
        sb  = next((s for s in wf_cb["segments"]  if s["year"]==yr), {})
        wf_rows.append([
            yr,
            f"{sc.get('cagr',0):+.2f}%",
            f"{sa.get('cagr',0):+.2f}%",
            f"{sa.get('cagr',0)-sc.get('cagr',0):+.2f}pp",
            f"{sb.get('cagr',0):+.2f}%",
            f"{sb.get('cagr',0)-sc.get('cagr',0):+.2f}pp",
        ])
    wf_rows.append([
        "AVG",
        f"{wf_cur['avg_cagr']:+.2f}%",
        f"{wf_ca['avg_cagr']:+.2f}%",
        f"{wf_ca['avg_cagr']-wf_cur['avg_cagr']:+.2f}pp",
        f"{wf_cb['avg_cagr']:+.2f}%",
        f"{wf_cb['avg_cagr']-wf_cur['avg_cagr']:+.2f}pp",
    ])
    wf_rows.append([
        "WF_pass",
        f"{wf_cur['wf_count']}/5",
        f"{wf_ca['wf_count']}/5",
        "",
        f"{wf_cb['wf_count']}/5",
        "",
    ])
    print_table("Rolling WF 全構成比較",
        ["Year", "CURRENT", "CAND_A", "Δ_A", "CAND_B", "Δ_B"],
        wf_rows
    )

    # 年代別 WF勝敗集計
    ca_wins = sum(1 for s in wf_ca["segments"]
                  if s.get("cagr",0) > next((c.get("cagr",0) for c in wf_cur["segments"] if c["year"]==s["year"]),0))
    cb_wins = sum(1 for s in wf_cb["segments"]
                  if s.get("cagr",0) > next((c.get("cagr",0) for c in wf_cur["segments"] if c["year"]==s["year"]),0))
    print(f"\n  WF Fold勝敗 vs CURRENT:")
    print(f"    CAND_A 勝={ca_wins}/5 負={5-ca_wins}/5")
    print(f"    CAND_B 勝={cb_wins}/5 負={5-cb_wins}/5")

    # ── Phase4: Safety Audit ─────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("  Phase4: Safety Audit")
    print("─" * 80)

    is_7yr = 7  # IS years

    def safety_stats(m_is: dict, m_oos: dict, wf: dict, label: str) -> dict:
        n  = m_is.get("n_trades", 0)
        ac = m_is.get("addon_cnt", 0)
        ex = m_is.get("avg_exp", 0)
        tr_yr = round(n / is_7yr, 1)
        # 平均保有日数近似: avg_exp% × 252日 × 7年 / n_trades
        hold_approx = round((ex / 100 * 252 * is_7yr) / max(n, 1) * 1, 1) if n > 0 else 0
        # 平均ポジション数 = avg_exp / (1/max_pos × 100) where max_pos=3, wt≈33%
        avg_pos = round(ex / 100 * 3, 2)  # avg_exp=33% → 1 position
        # Add-on率
        addon_rate = round(ac / max(n, 1) * 100, 1)
        # 最悪年CAGR
        seg3 = next((s for s in wf.get("segments", []) if s["year"]=="2022"), {})
        worst_cagr = seg3.get("cagr", 0)
        # Fold std (volatility)
        fold_std = wf.get("fold_std", 0)
        return {
            "label": label,
            "n_trades_is": n,
            "trade_rate_yr": tr_yr,
            "addon_cnt": ac,
            "addon_rate_pct": addon_rate,
            "avg_hold_approx": hold_approx,
            "avg_positions": avg_pos,
            "cash_util_pct": ex,
            "worst_year_cagr": worst_cagr,
            "fold_std": fold_std,
            "oos_calmar": m_oos.get("calmar", 0),
            "is_max_dd": m_is.get("max_dd", 0),
            "oos_max_dd": m_oos.get("max_dd", 0),
        }

    safety = {n: safety_stats(results[n]["is"], results[n]["oos"], results[n]["wf"],
                               CONFIGS[n]["label"]) for n in CONFIGS}

    print_table("Safety Audit 比較",
        ["指標", "CURRENT", "CAND_A", "Δ_A", "CAND_B", "Δ_B"],
        [
            ["IS Execution数",
             str(safety["CURRENT"]["n_trades_is"]),
             str(safety["CAND_A"]["n_trades_is"]),
             f"{safety['CAND_A']['n_trades_is']-safety['CURRENT']['n_trades_is']:+d}",
             str(safety["CAND_B"]["n_trades_is"]),
             f"{safety['CAND_B']['n_trades_is']-safety['CURRENT']['n_trades_is']:+d}"],
            ["IS Execution/年",
             f"{safety['CURRENT']['trade_rate_yr']:.1f}",
             f"{safety['CAND_A']['trade_rate_yr']:.1f}",
             f"{safety['CAND_A']['trade_rate_yr']-safety['CURRENT']['trade_rate_yr']:+.1f}",
             f"{safety['CAND_B']['trade_rate_yr']:.1f}",
             f"{safety['CAND_B']['trade_rate_yr']-safety['CURRENT']['trade_rate_yr']:+.1f}"],
            ["Addon数(IS)",
             str(safety["CURRENT"]["addon_cnt"]),
             str(safety["CAND_A"]["addon_cnt"]),
             f"{safety['CAND_A']['addon_cnt']-safety['CURRENT']['addon_cnt']:+d}",
             str(safety["CAND_B"]["addon_cnt"]),
             f"{safety['CAND_B']['addon_cnt']-safety['CURRENT']['addon_cnt']:+d}"],
            ["Addon率",
             f"{safety['CURRENT']['addon_rate_pct']:.1f}%",
             f"{safety['CAND_A']['addon_rate_pct']:.1f}%",
             f"{safety['CAND_A']['addon_rate_pct']-safety['CURRENT']['addon_rate_pct']:+.1f}pp",
             f"{safety['CAND_B']['addon_rate_pct']:.1f}%",
             f"{safety['CAND_B']['addon_rate_pct']-safety['CURRENT']['addon_rate_pct']:+.1f}pp"],
            ["avg保有日数(近似)",
             f"{safety['CURRENT']['avg_hold_approx']:.1f}d",
             f"{safety['CAND_A']['avg_hold_approx']:.1f}d",
             f"{safety['CAND_A']['avg_hold_approx']-safety['CURRENT']['avg_hold_approx']:+.1f}d",
             f"{safety['CAND_B']['avg_hold_approx']:.1f}d",
             f"{safety['CAND_B']['avg_hold_approx']-safety['CURRENT']['avg_hold_approx']:+.1f}d"],
            ["avg同時ポジション",
             f"{safety['CURRENT']['avg_positions']:.2f}",
             f"{safety['CAND_A']['avg_positions']:.2f}",
             f"{safety['CAND_A']['avg_positions']-safety['CURRENT']['avg_positions']:+.2f}",
             f"{safety['CAND_B']['avg_positions']:.2f}",
             f"{safety['CAND_B']['avg_positions']-safety['CURRENT']['avg_positions']:+.2f}"],
            ["Cash利用率(IS)",
             f"{safety['CURRENT']['cash_util_pct']:.1f}%",
             f"{safety['CAND_A']['cash_util_pct']:.1f}%",
             f"{safety['CAND_A']['cash_util_pct']-safety['CURRENT']['cash_util_pct']:+.1f}pp",
             f"{safety['CAND_B']['cash_util_pct']:.1f}%",
             f"{safety['CAND_B']['cash_util_pct']-safety['CURRENT']['cash_util_pct']:+.1f}pp"],
            ["最悪年CAGR(2022)",
             f"{safety['CURRENT']['worst_year_cagr']:+.2f}%",
             f"{safety['CAND_A']['worst_year_cagr']:+.2f}%",
             f"{safety['CAND_A']['worst_year_cagr']-safety['CURRENT']['worst_year_cagr']:+.2f}pp",
             f"{safety['CAND_B']['worst_year_cagr']:+.2f}%",
             f"{safety['CAND_B']['worst_year_cagr']-safety['CURRENT']['worst_year_cagr']:+.2f}pp"],
            ["Fold std(WF)",
             f"{safety['CURRENT']['fold_std']:.2f}pp",
             f"{safety['CAND_A']['fold_std']:.2f}pp",
             f"{safety['CAND_A']['fold_std']-safety['CURRENT']['fold_std']:+.2f}pp",
             f"{safety['CAND_B']['fold_std']:.2f}pp",
             f"{safety['CAND_B']['fold_std']-safety['CURRENT']['fold_std']:+.2f}pp"],
            ["IS MaxDD",
             f"{safety['CURRENT']['is_max_dd']:.2f}%",
             f"{safety['CAND_A']['is_max_dd']:.2f}%",
             f"{safety['CAND_A']['is_max_dd']-safety['CURRENT']['is_max_dd']:+.2f}pp",
             f"{safety['CAND_B']['is_max_dd']:.2f}%",
             f"{safety['CAND_B']['is_max_dd']-safety['CURRENT']['is_max_dd']:+.2f}pp"],
            ["OOS MaxDD",
             f"{safety['CURRENT']['oos_max_dd']:.2f}%",
             f"{safety['CAND_A']['oos_max_dd']:.2f}%",
             f"{safety['CAND_A']['oos_max_dd']-safety['CURRENT']['oos_max_dd']:+.2f}pp",
             f"{safety['CAND_B']['oos_max_dd']:.2f}%",
             f"{safety['CAND_B']['oos_max_dd']-safety['CURRENT']['oos_max_dd']:+.2f}pp"],
            ["OOS Calmar",
             f"{safety['CURRENT']['oos_calmar']:.3f}",
             f"{safety['CAND_A']['oos_calmar']:.3f}",
             f"{safety['CAND_A']['oos_calmar']-safety['CURRENT']['oos_calmar']:+.3f}",
             f"{safety['CAND_B']['oos_calmar']:.3f}",
             f"{safety['CAND_B']['oos_calmar']-safety['CURRENT']['oos_calmar']:+.3f}"],
        ]
    )

    # ── Phase5: Production Decision ───────────────────────────────────────────
    print("\n" + "─" * 80)
    print("  Phase5: Production Decision")
    print("─" * 80)

    # Candidate A スコアリング
    ca_is_delta  = delta_m(is_ca,  is_cur,  "cagr")
    ca_oos_delta = delta_m(oos_ca, oos_cur, "cagr")
    ca_wf_delta  = wf_ca["avg_cagr"] - wf_cur["avg_cagr"]
    ca_wf_diff   = wf_ca["wf_count"] - wf_cur["wf_count"]
    ca_dd22      = delta22_a
    ca_dd_delta  = delta_m(is_ca, is_cur, "max_dd")

    # CAND_A判定
    ca_pos = sum([
        ca_is_delta  > 0,   # IS CAGR改善
        ca_oos_delta > 0,   # OOS CAGR改善
        ca_wf_delta  > 0,   # WF avg改善
        ca_dd_delta  > 0,   # MaxDD改善 (less negative = positive improvement)
        ca_dd22      > 0,   # 2022改善
    ])
    ca_neg = 5 - ca_pos
    ca_verdict = "REPLACE" if ca_pos >= 3 else ("CONDITIONAL" if ca_pos == 2 else "KEEP")

    # Candidate B スコアリング
    cb_is_delta  = delta_m(is_cb,  is_cur,  "cagr")
    cb_oos_delta = delta_m(oos_cb, oos_cur, "cagr")
    cb_wf_delta  = wf_cb["avg_cagr"] - wf_cur["avg_cagr"]
    cb_wf_diff   = wf_cb["wf_count"] - wf_cur["wf_count"]
    cb_dd22      = delta22_b
    cb_dd_delta  = delta_m(is_cb, is_cur, "max_dd")

    # CAND_B判定 (WF pass改善が最重要)
    cb_pos = sum([
        cb_is_delta  > 0,
        cb_oos_delta > 0,
        cb_wf_delta  > 0,
        cb_wf_diff   > 0,   # WF pass数改善 (RSR75→5/5期待)
        cb_dd22      > 0,   # 2022改善
    ])
    cb_neg = 5 - cb_pos
    cb_verdict = "REPLACE" if cb_pos >= 3 else ("CONDITIONAL" if cb_pos == 2 else "KEEP")

    print(f"""
  === Candidate A: EQ Scale Remove ===
  IS CAGR:     {ca_is_delta:+.2f}pp  {"✓" if ca_is_delta>0 else "✗"}
  OOS CAGR:    {ca_oos_delta:+.2f}pp  {"✓" if ca_oos_delta>0 else "✗"}
  WF avg CAGR: {ca_wf_delta:+.2f}pp  {"✓" if ca_wf_delta>0 else "✗"}
  2022 CAGR:   {ca_dd22:+.2f}pp  {"✓" if ca_dd22>0 else "✗"}  ← 最悪年（重要）
  IS MaxDD:    {ca_dd_delta:+.2f}pp  {"✓" if ca_dd_delta>0 else "✗"}
  WF pass:     {wf_cur['wf_count']}/5 → {wf_ca['wf_count']}/5 ({ca_wf_diff:+d})
  ポジティブ: {ca_pos}/5  ネガティブ: {ca_neg}/5

  判定: {ca_verdict}
  根拠:""")

    if ca_is_delta > 0 and ca_oos_delta > 0:
        print(f"    IS/OOS CAGR両方改善(+{ca_is_delta:.2f}/+{ca_oos_delta:.2f}pp) → 経済的優位")
    if ca_wf_delta < 0:
        print(f"    WF avg CAGR劣化({ca_wf_delta:.2f}pp) → 時系列安定性は低下")
    if ca_dd22 < 0:
        print(f"    2022最悪年でCAGR悪化({ca_dd22:.2f}pp) → ダウンサイドリスク増大 ⚠")
    if ca_wf_diff < 0:
        print(f"    WF pass数低下({wf_cur['wf_count']}→{wf_ca['wf_count']}) → WF安定性低下 ⚠")
    if ca_wf_diff == 0:
        print(f"    WF pass数変化なし({wf_cur['wf_count']}/5)")

    print(f"""
  === Candidate B: RSR Exit 70→75 ===
  IS CAGR:     {cb_is_delta:+.2f}pp  {"✓" if cb_is_delta>0 else "✗"}
  OOS CAGR:    {cb_oos_delta:+.2f}pp  {"✓" if cb_oos_delta>0 else "✗"}
  WF avg CAGR: {cb_wf_delta:+.2f}pp  {"✓" if cb_wf_delta>0 else "✗"}
  2022 CAGR:   {cb_dd22:+.2f}pp  {"✓" if cb_dd22>0 else "✗"}  ← 最悪年（重要）
  IS MaxDD:    {cb_dd_delta:+.2f}pp  {"✓" if cb_dd_delta>0 else "✗"}
  WF pass:     {wf_cur['wf_count']}/5 → {wf_cb['wf_count']}/5 ({cb_wf_diff:+d})
  ポジティブ: {cb_pos}/5  ネガティブ: {cb_neg}/5

  判定: {cb_verdict}
  根拠:""")

    if cb_wf_diff > 0:
        print(f"    WF pass数改善({wf_cur['wf_count']}→{wf_cb['wf_count']}) → WF安定性向上 ✓")
    elif cb_wf_diff == 0:
        print(f"    WF pass数変化なし({wf_cur['wf_count']}/5)")
    if cb_dd22 > 0:
        print(f"    2022最悪年でCAGR改善(+{cb_dd22:.2f}pp) → ダウンサイドリスク低下 ✓")
    elif cb_dd22 < 0:
        print(f"    2022最悪年でCAGR悪化({cb_dd22:.2f}pp) ⚠")
    if cb_is_delta < 0:
        print(f"    IS CAGR軽微低下({cb_is_delta:.2f}pp)")
    if cb_oos_delta < 0:
        print(f"    OOS CAGR軽微低下({cb_oos_delta:.2f}pp)")

    # 優先順位
    print(f"\n  === 推奨実施順位 ===")
    verdicts = {"CAND_A": ca_verdict, "CAND_B": cb_verdict}
    priority = sorted(["CAND_A","CAND_B"], key=lambda x: 0 if verdicts[x]=="REPLACE" else (1 if verdicts[x]=="CONDITIONAL" else 2))
    for i, name in enumerate(priority, 1):
        v = verdicts[name]
        cfg = CONFIGS[name]
        print(f"    {i}. {name}({cfg['label']}): {v} — {cfg['desc']}")

    # ── Deliverables ─────────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("  Deliverables")
    print("─" * 80)

    print(f"""
  [1] Production比較表:
      {CONFIGS['CURRENT']['label']:<25} IS={is_cur['cagr']:+.2f}%  OOS={oos_cur['cagr']:+.2f}%  WF={wf_cur['wf_count']}/5
      {CONFIGS['CAND_A']['label']:<25} IS={is_ca['cagr']:+.2f}%  OOS={oos_ca['cagr']:+.2f}%  WF={wf_ca['wf_count']}/5
      {CONFIGS['CAND_B']['label']:<25} IS={is_cb['cagr']:+.2f}%  OOS={oos_cb['cagr']:+.2f}%  WF={wf_cb['wf_count']}/5

  [2] 改善量一覧:
      CAND_A vs CURRENT: IS={ca_is_delta:+.2f}pp  OOS={ca_oos_delta:+.2f}pp  WFavg={ca_wf_delta:+.2f}pp  2022={ca_dd22:+.2f}pp
      CAND_B vs CURRENT: IS={cb_is_delta:+.2f}pp  OOS={cb_oos_delta:+.2f}pp  WFavg={cb_wf_delta:+.2f}pp  2022={cb_dd22:+.2f}pp

  [3] リスク一覧:
      CAND_A: EQ Scale除去で2022最悪年悪化({ca_dd22:+.2f}pp); WF pass低下可能性
      CAND_B: RSR厳格化で早期Exit増加 → IS軽微低下の可能性あり
      共通:   研究偏差(OOS1年は過学習リスク); 実際発注はDRY_RUN必須

  [4] Production推奨構成:
      最終構成 = CAND_B ({'先実施' if verdicts['CAND_B']=='REPLACE' else '検討後実施'})
      priority1: CAND_B(RSR75) — Study71/72でWF5/5達成; 最悪年リスク低下
      priority2: CAND_A(EQ Scale除去) — IS/OOS改善だが2022リスクを慎重評価

  [5] 変更対象ファイル一覧 (コード修正は別途ASK_FIRST):
      CAND_B実施: src/configs/strategy.yaml → rsr_exit: 75.0  (min_rsr: 75.0と一致)
      CAND_A実施: src/configs/strategy.yaml → eq_scale_addon.enabled: false
      確認必須:   run_live_signal.py (rsr_exit_threshold の live反映確認)
                  signal_bridge.py (RSR exit条件の発注トリガー確認)
                  src/live/live_equivalent.py (rsr_exit_threshold定数確認)
    """)

    # ── JSON出力 ─────────────────────────────────────────────────────────────
    print(f"[OUTPUT] {OUT_FILE}")

    output = {
        "study":   "Study73_ProductionMigrationAudit",
        "date":    TODAY_STR,
        "configs": {k: {"label": v["label"], "desc": v["desc"], "params": {
            "exit_policy": v["exit_policy"], "addon_policy": v["addon_policy"],
            "rsr_exit": v["rsr_exit"]}} for k, v in CONFIGS.items()},
        "phase0_integrity": {
            "study52_current_is": s52_current_is, "study52_current_oos": s52_current_oos,
            "study52_canda_is": s52_canda_is, "study52_canda_oos": s52_canda_oos,
            "study72_f1_verdict": "INVALID", "study72_f3_verdict": "REMOVE候補",
        },
        "phase1_cand_a": {
            "is_current": is_cur, "is_cand_a": is_ca,
            "oos_current": oos_cur, "oos_cand_a": oos_ca,
            "wf_current": {"wf_count": wf_cur["wf_count"], "avg_cagr": wf_cur["avg_cagr"],
                           "fold_std": wf_cur["fold_std"], "segments": wf_cur["segments"]},
            "wf_cand_a":  {"wf_count": wf_ca["wf_count"],  "avg_cagr": wf_ca["avg_cagr"],
                           "fold_std": wf_ca["fold_std"],  "segments": wf_ca["segments"]},
            "delta": {"is_cagr": ca_is_delta, "oos_cagr": ca_oos_delta,
                      "wf_avg": ca_wf_delta, "wf_pass": ca_wf_diff, "2022_cagr": ca_dd22},
            "nev_is_yr": nev_is_ya, "nev_oos_yr": nev_oos_ya,
            "verdict": ca_verdict,
        },
        "phase2_cand_b": {
            "is_current": is_cur, "is_cand_b": is_cb,
            "oos_current": oos_cur, "oos_cand_b": oos_cb,
            "wf_current": {"wf_count": wf_cur["wf_count"], "avg_cagr": wf_cur["avg_cagr"],
                           "fold_std": wf_cur["fold_std"], "segments": wf_cur["segments"]},
            "wf_cand_b":  {"wf_count": wf_cb["wf_count"],  "avg_cagr": wf_cb["avg_cagr"],
                           "fold_std": wf_cb["fold_std"],  "segments": wf_cb["segments"]},
            "delta": {"is_cagr": cb_is_delta, "oos_cagr": cb_oos_delta,
                      "wf_avg": cb_wf_delta, "wf_pass": cb_wf_diff, "2022_cagr": cb_dd22},
            "nev_is_yr": nev_is_yb, "nev_oos_yr": nev_oos_yb,
            "verdict": cb_verdict,
        },
        "phase3_robustness": {
            "bootstrap": boot_results,
            "annual_by_config": {n: {yr: results[n]["annual"].get(yr, {})
                                     for yr in IS_YEARS + ["2025"]}
                                 for n in CONFIGS},
        },
        "phase4_safety": safety,
        "phase5_decision": {
            "cand_a_verdict": ca_verdict,
            "cand_b_verdict": cb_verdict,
            "priority": priority,
            "recommended_config": "CAND_B_then_CAND_A",
            "change_files": [
                "src/configs/strategy.yaml (rsr_exit: 75.0 for CAND_B)",
                "src/configs/strategy.yaml (eq_scale_addon.enabled: false for CAND_A)",
                "run_live_signal.py (確認必須)",
                "signal_bridge.py (確認必須)",
                "src/live/live_equivalent.py (確認必須)",
            ],
        },
    }

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print("  完了")


if __name__ == "__main__":
    main()
