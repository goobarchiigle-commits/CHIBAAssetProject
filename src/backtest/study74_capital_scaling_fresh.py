"""
study74_capital_scaling_fresh.py
Study74 — 資本スケーリング清浄再検証（M1適用後エンジン・プログラム最優先）

正典定義: 成功=¥20-30MでCAGR≥22%（fix後・コスト込・WF5/5）。失敗=<18% or DD%が資本比例悪化。
終了条件: 4資本点×2構成の全測定（追加スイープ禁止）。

追加仕様（2026-07-04 ユーザー拡張指示）:
  Part A: 資本制約の分解（waterfall寄与分析）— lot丸め/max_positions/symbol_capを
          既存の研究用レバー（lot_size/max_positions_override/risk_controls.symbol_cap）
          で1つずつ解除し、各々のΔCAGR寄与を測定。新規エンジン改修なし。
  Part B: Capacity分析 — スキップ率/平均投資率/現金滞留率/lot不足率/Position充足率を
          資本水準別に可視化（全て既存計装の再利用）。

禁止: PARAMS_LOCKED変更 / Production変更 / ¥3M固定でのProduction提案（Part Aは診断専用ツール）。
汚染前Study42/43A/46のJSONは参照値としてのみ使用・判定に使用禁止。
"""
from __future__ import annotations

import json
import sys
import warnings
import dataclasses
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
OUT_FILE = ROOT / "backtests" / f"study74_capital_scaling_{TODAY_STR}.json"

MIN_HOLD = 3
IS_START, IS_END = "2018-01-01", "2024-12-31"
OOS_START, OOS_END = "2025-01-01", "2025-12-31"
DATA_END = "2025-12-31"
YEARS_STANDALONE = [("2018", "2018-01-01", "2018-12-31"), ("2019", "2019-01-01", "2019-12-31")]

WF_SEGS = [
    {"oos_s": "2020-01-01", "oos_e": "2020-12-31", "year": "2020"},
    {"oos_s": "2021-01-01", "oos_e": "2021-12-31", "year": "2021"},
    {"oos_s": "2022-01-01", "oos_e": "2022-12-31", "year": "2022"},
    {"oos_s": "2023-01-01", "oos_e": "2023-12-31", "year": "2023"},
    {"oos_s": "2024-01-01", "oos_e": "2024-12-31", "year": "2024"},
]

CAPITAL_LEVELS = [3_000_000, 10_000_000, 20_000_000, 30_000_000]
CONFIGS = {
    "CURRENT": {"rsr_exit": 70.0, "label": "D_ATR_EQ(RSR70)"},
    "CAND_B":  {"rsr_exit": 75.0, "label": "D_ATR_EQ(RSR75)"},
}


def get_active(ds, all_syms, start, end):
    from src.config_loader import load_strategy_config
    cfg = load_strategy_config()
    bc = cfg.risk_controls.bear_universe_filter
    be = list(bc.excluded_sectors) if bc.enabled else None
    return cab.build_dyn_rsr42_active(
        universe_raw=ds["universe_raw"], topix_close=ds["topix_close"],
        rsr_df=ds["rsr_df"], all_syms=all_syms, start=start, end=end,
        bear_exclude_sectors=be,
        sym_sector_map=dict(ds["trade_syms"]) if be else None,
    )


def run_bt(ds, sym_active_df, start, end, capital, rsr_exit=70.0,
           lot_size=100, max_positions_override=None, cfg_obj=None):
    return cab.run_scenario(
        scenario="BASELINE",
        universe_raw=ds["universe_raw"], rsr_df=ds["rsr_df"], alpha_df=None,
        regime_df=ds["regime_df"], trade_syms=ds["trade_syms"], rsr_syms=ds["rsr_syms"],
        cfg=cfg_obj if cfg_obj is not None else ds["base_cfg"],
        start=start, end=end, verbose=False,
        tech_matrices=ds["tech_matrices"], breadth_series=ds["breadth_series"],
        capital=capital, min_hold=MIN_HOLD, topix_close=ds["topix_close"],
        market_shock_mode="composite", rsr_exit_threshold=rsr_exit,
        sym_active_df=sym_active_df,
        enable_simple_rsr_exit=True, enable_atr_trailing_prod=True,
        enable_multilayer_rsr=True, enable_atr_risk_sizing=False, enable_mtf_filter=False,
        sizing_mode="existing", exit_policy="A", addon_policy="D",
        addon_size_frac=0.25, addon_atr_mult=1.0,
        lot_size=lot_size, max_positions_override=max_positions_override,
    )


def safe_float(v, default=0.0):
    try:
        f = float(v); return f if not np.isnan(f) else default
    except (TypeError, ValueError): return default


def extract_metrics(raw: dict) -> dict:
    n_cand_total = 0  # placeholder; avg_candidates×n_daysで近似
    return {
        "cagr": round(safe_float(raw.get("cagr")), 2),
        "sharpe": round(safe_float(raw.get("sharpe")), 3),
        "max_dd": round(safe_float(raw.get("max_dd")), 2),
        "calmar": round(safe_float(raw.get("calmar")), 3),
        "n_trades": int(raw.get("n_trades", 0) or 0),
        "avg_exp": round(safe_float(raw.get("avg_exposure", 0)), 1),
        "avg_candidates": round(safe_float(raw.get("avg_candidates", 0)), 2),
        "avg_simultaneous_holdings": round(safe_float(raw.get("avg_simultaneous_holdings", 0)), 2),
        "rejected_by_lot_count": int(raw.get("rejected_by_lot_count", 0) or 0),
        "missed_by_cap_count": int(raw.get("missed_by_cap_count", 0) or 0),
        "avg_idle_cash_ratio_pct": round(safe_float(raw.get("avg_idle_cash_ratio_pct", 0)), 1),
        "q1_idle_when_winner_pct": raw.get("q1_idle_when_winner_pct"),
    }


def run_wf(ds, all_syms, capital, rsr_exit, label="") -> dict:
    seg_rows, cagrs, pass_cnt = [], [], 0
    for fold in WF_SEGS:
        act = get_active(ds, all_syms, fold["oos_s"], fold["oos_e"])
        m = extract_metrics(run_bt(ds, act, fold["oos_s"], fold["oos_e"], capital, rsr_exit=rsr_exit))
        wf_pass = m["cagr"] > 0
        if wf_pass: pass_cnt += 1
        cagrs.append(m["cagr"])
        seg_rows.append({**m, "year": fold["year"], "wf_pass": wf_pass})
        print(f"      [{label}] WF {fold['year']}: CAGR={m['cagr']:+.2f}%  {'✓' if wf_pass else '✗'}")
    avg = round(float(np.mean(cagrs)), 2) if cagrs else 0.0
    return {"wf_count": pass_cnt, "avg_cagr": avg, "segments": seg_rows}


def run_annual_standalone(ds, all_syms, capital, rsr_exit, label="") -> dict:
    annual = {}
    for yr, start, end in YEARS_STANDALONE:
        act = get_active(ds, all_syms, start, end)
        try:
            m = extract_metrics(run_bt(ds, act, start, end, capital, rsr_exit=rsr_exit))
            annual[yr] = m
        except Exception as e:
            annual[yr] = {"cagr": 0.0, "error": str(e)}
    return annual


# ======================================================================
# Part A: 資本制約の分解（waterfall）
# ======================================================================

def constraint_waterfall(ds, all_syms, capital) -> dict:
    act_is = get_active(ds, all_syms, IS_START, IS_END)
    variants = {
        "baseline":         dict(lot_size=100, max_positions_override=None, cfg_obj=None),
        "relax_lot":        dict(lot_size=1,   max_positions_override=None, cfg_obj=None),
        "relax_maxpos":     dict(lot_size=100, max_positions_override=10,   cfg_obj=None),
        "relax_symcap":     dict(lot_size=100, max_positions_override=None, cfg_obj="SYMCAP"),
        "relax_all":        dict(lot_size=1,   max_positions_override=10,   cfg_obj="SYMCAP"),
    }
    results = {}
    for name, kw in variants.items():
        cfg_obj = None
        if kw["cfg_obj"] == "SYMCAP":
            new_rc = dataclasses.replace(ds["base_cfg"].risk_controls, symbol_cap=1.0)
            cfg_obj = dataclasses.replace(ds["base_cfg"], risk_controls=new_rc)
        raw = run_bt(ds, act_is, IS_START, IS_END, capital, rsr_exit=70.0,
                     lot_size=kw["lot_size"], max_positions_override=kw["max_positions_override"],
                     cfg_obj=cfg_obj)
        m = extract_metrics(raw)
        results[name] = m
        print(f"    [capital={capital/1e6:.0f}M][{name}] IS CAGR={m['cagr']:+.2f}%  MaxDD={m['max_dd']:.2f}%  Trades={m['n_trades']}")

    base_cagr = results["baseline"]["cagr"]
    delta_lot = round(results["relax_lot"]["cagr"] - base_cagr, 2)
    delta_maxpos = round(results["relax_maxpos"]["cagr"] - base_cagr, 2)
    delta_symcap = round(results["relax_symcap"]["cagr"] - base_cagr, 2)
    delta_all = round(results["relax_all"]["cagr"] - base_cagr, 2)
    sum_individual = round(delta_lot + delta_maxpos + delta_symcap, 2)
    interaction = round(delta_all - sum_individual, 2)

    return {
        "capital": capital, "variants": results,
        "cagr_drag_pp": {
            "lot_rounding": -delta_lot if delta_lot < 0 else 0.0,
            "max_positions": -delta_maxpos if delta_maxpos < 0 else 0.0,
            "symbol_cap": -delta_symcap if delta_symcap < 0 else 0.0,
        },
        "delta_cagr": {"lot": delta_lot, "max_positions": delta_maxpos, "symbol_cap": delta_symcap, "all_combined": delta_all},
        "sum_of_individual_deltas": sum_individual, "interaction_effect_pp": interaction,
    }


# ======================================================================
# メイン
# ======================================================================

def main():
    print("=" * 80)
    print("  Study74 — 資本スケーリング清浄再検証 + 制約分解 + Capacity分析")
    print(f"  Date: {TODAY_STR}   (M1適用後エンジン)")
    print("=" * 80)

    ds = build_common_dataset(DATA_END)
    all_syms = list(ds["trade_syms"].keys())
    print(f"  {len(all_syms)} symbols")

    # ── 基本マトリクス: capital × config ──────────────────────────────────
    matrix = {}
    for capital in CAPITAL_LEVELS:
        matrix[str(capital)] = {}
        for cname, cfg_p in CONFIGS.items():
            label = f"{capital/1e6:.0f}M-{cname}"
            print(f"\n[RUN] capital=¥{capital:,} config={cname}({cfg_p['label']})")

            act_is = get_active(ds, all_syms, IS_START, IS_END)
            m_is = extract_metrics(run_bt(ds, act_is, IS_START, IS_END, capital, rsr_exit=cfg_p["rsr_exit"]))
            print(f"    IS:  CAGR={m_is['cagr']:+.2f}%  MaxDD={m_is['max_dd']:.2f}%  Trades={m_is['n_trades']}  "
                  f"AvgExp={m_is['avg_exp']:.1f}%  LotReject={m_is['rejected_by_lot_count']}  CapMiss={m_is['missed_by_cap_count']}")

            act_oos = get_active(ds, all_syms, OOS_START, OOS_END)
            m_oos = extract_metrics(run_bt(ds, act_oos, OOS_START, OOS_END, capital, rsr_exit=cfg_p["rsr_exit"]))
            print(f"    OOS: CAGR={m_oos['cagr']:+.2f}%  MaxDD={m_oos['max_dd']:.2f}%  Trades={m_oos['n_trades']}")

            wf = run_wf(ds, all_syms, capital, cfg_p["rsr_exit"], label=label)
            annual_sa = run_annual_standalone(ds, all_syms, capital, cfg_p["rsr_exit"], label=label)

            matrix[str(capital)][cname] = {"is": m_is, "oos": m_oos, "wf": wf, "annual_standalone": annual_sa}

    # ── Part A: 制約waterfall（CURRENT構成のみ・各資本水準） ────────────────
    print("\n" + "─" * 80)
    print("  Part A: 資本制約の分解（waterfall・CURRENT構成・IS期間）")
    print("─" * 80)
    waterfalls = {}
    for capital in CAPITAL_LEVELS:
        waterfalls[str(capital)] = constraint_waterfall(ds, all_syms, capital)

    # ── Part B: Capacity分析（CURRENT構成の基本マトリクスから抽出） ─────────
    print("\n" + "─" * 80)
    print("  Part B: Capacity分析（CURRENT構成・IS期間）")
    print("─" * 80)
    capacity = {}
    for capital in CAPITAL_LEVELS:
        m = matrix[str(capital)]["CURRENT"]["is"]
        total_candidates = m["avg_candidates"] * 1761  # IS 2018-2024 ≈ 1761営業日（近似）
        lot_reject = m["rejected_by_lot_count"]
        cap_miss = m["missed_by_cap_count"]
        skip_rate = round((lot_reject + cap_miss) / max(1, total_candidates) * 100, 2)
        lot_shortage_rate = round(lot_reject / max(1, total_candidates) * 100, 2)
        position_fill_rate = round(m["avg_simultaneous_holdings"] / 3.0 * 100, 1)
        capacity[str(capital)] = {
            "skip_rate_pct": skip_rate, "avg_investment_ratio_pct": m["avg_exp"],
            "cash_idle_ratio_pct": m["avg_idle_cash_ratio_pct"],
            "lot_shortage_rate_pct": lot_shortage_rate, "position_fill_rate_pct": position_fill_rate,
            "rejected_by_lot_count": lot_reject, "missed_by_cap_count": cap_miss,
        }
        print(f"  capital=¥{capital/1e6:.0f}M: skip={skip_rate}%  投資率={m['avg_exp']}%  "
              f"現金滞留={m['avg_idle_cash_ratio_pct']}%  lot不足={lot_shortage_rate}%  Position充足={position_fill_rate}%")

    output = {
        "study": "Study74_capital_scaling", "date": TODAY_STR,
        "engine": "M1適用後(addon=open執行)/M2変更なし",
        "capital_config_matrix": matrix,
        "part_a_constraint_waterfall": waterfalls,
        "part_b_capacity_analysis": capacity,
    }
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[OUTPUT] {OUT_FILE}")


if __name__ == "__main__":
    main()
