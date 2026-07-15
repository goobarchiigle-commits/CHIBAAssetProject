"""
src/backtest/study76_datr_eq_universe_c_rebaseline.py
D_ATR_EQ Study75-Universe再ベースライン（Study76前提工程・「Dynamic RSR42」測定）。

canon上の位置づけ: reports/study76_execution_plan.md が定義する「Study76」は
Clenow純正ベンチマーク（D_ATR_EQを全面簡略化する別実験）であり、本スクリプトはそれとは別物。
本スクリプトは Study76 が比較対象として必要とする前提工程
「D_ATR_EQをStudy75 Universe上でfresh run再測定する」を実装する
（study76_execution_plan.md §3/§5・study76_dependency_matrix.md §2）。

設計思想（詳細は plan 参照・要点のみ）:
  D_ATR_EQのアーキテクチャ（Exit/リスク/breadth/dyn_rsr42_bear_rs0）は一切変更しない。
  hindsight静的RSR42リストだけを、Study75AのUniverse C（月次PIT規則ユニバース）から
  各月T-1時点のトレイリング・コンポジットリターン上位42銘柄を機械的に選ぶ
  「Dynamic RSR42」（PIT・rule-based・月次固定42名ローテーション）に置き換える。
  選ばれた42名の中でのみRSRパーセンタイル・min_rsr>=75・dyn_rsr42_bear_rs0のTop30/Bear20を
  計算する（本番と全く同じ解像度）ため、Study75Bが発見したプールサイズ相対性の歪みを
  エンジン無改変のまま回避できる。

RunB（Dynamic RSR42・本スクリプトの新規fresh run・主分析対象）とRunA（Universe C全体へ
直接パーセンタイルRSR適用・Negative Control）のうちRunAは既存のStudy75B U3
（backtests/study75_survivorship_2026-07-11.json）をそのまま参照する（再実行しない・
U3は percentile歪み+セクターキャップ崩壊の二重汚染が判明済みで、そもそも参考記録専用の
位置づけとして丁度良い）。

最重要指標: Δ_dynamic = RunB(Dynamic RSR42) − U0(静的hindsight RSR42・Study75B既存値)。
"""
from __future__ import annotations

import sys
import warnings
from collections import Counter

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import src.backtest.composite_alpha_bt as cab
from src.backtest.rsr import calc_composite_return
from src.backtest.study75b_survivorship_bias import (
    IS_END, IS_START, OOS_END, OOS_START,
    CAPITAL, MIN_HOLD,
    build_scenario_dataset, run_bt, window_dataset, and_masks, summarize,
    load_universe_c, load_panel,
)
from src.config_loader import load_strategy_config
from src.paths import DATABASE_MASTER_DIR, RESULTS_DIR
from src.strategy.universe import build_dyn_rsr42_active

_JST = timezone(timedelta(hours=9))

FIXED_N = 42  # RSR42と同一件数（他の全ロック定数がこのプールサイズを前提に較正されているため）
FULL_START, FULL_END = IS_START, OOS_END

WF_SEGS = [
    {"oos_s": "2020-01-01", "oos_e": "2020-12-31", "year": "2020"},
    {"oos_s": "2021-01-01", "oos_e": "2021-12-31", "year": "2021"},
    {"oos_s": "2022-01-01", "oos_e": "2022-12-31", "year": "2022"},
    {"oos_s": "2023-01-01", "oos_e": "2023-12-31", "year": "2023"},
    {"oos_s": "2024-01-01", "oos_e": "2024-12-31", "year": "2024"},
]

# 既存ベースライン値（再計算不要・そのまま引用。plan §1参照）
U0_REFERENCE = {
    "source": "backtests/study75_survivorship_2026-07-11.json scenario_summary.U0",
    "IS": {"cagr": 8.71, "sharpe": 0.677, "calmar": 0.421, "max_dd": -20.72, "n_trades": 279,
           "avg_exposure": 43.8, "avg_simultaneous_holdings": 1.99},
    "OOS": {"cagr": -0.98, "sharpe": -0.027, "calmar": -0.097, "max_dd": -10.07, "n_trades": 40,
            "avg_exposure": 38.1, "avg_simultaneous_holdings": 1.63},
}
OFFICIAL_PRODUCTION_REFERENCE = {
    "source": "backtests/study_m1_production_update_2026-07-04.json results.CURRENT (yfinance basis)",
    "IS": {"cagr": 12.22, "sharpe": 0.579, "calmar": 0.671, "max_dd": -18.22, "n_trades": 263,
           "avg_exposure": 31.5},
    "OOS": {"cagr": 11.42, "sharpe": 1.011, "calmar": 1.103, "max_dd": -10.35, "n_trades": 42,
            "avg_exposure": 36.4},
}
NEGATIVE_CONTROL_RUNA_REFERENCE = {
    "source": "backtests/study75_survivorship_2026-07-11.json scenario_summary.U3 "
              "(Universe C union(3020) rank pool, direct percentile RSR, monthly membership AND-mask)",
    "note": "既知の二重汚染（パーセンタイル距離歪み + 全銘柄'不明'セクターによるセクターキャップ崩壊、"
            "study75c_interpretation.md §6.0）。主結論・selection bias推定には使用しない。反証専用参照値。",
    "IS_cagr_approx": -30.60,
}


# ────────────────────────────────────────────────────────────────────────── #
# 1) Dynamic RSR42 構築（PIT・rule-based・月次固定42名選抜）
# ────────────────────────────────────────────────────────────────────────── #
def build_dynamic_rsr42_membership(
    monthly_universe: dict[str, list[str]],
    composite_scores: dict[str, pd.Series],
    fixed_n: int = FIXED_N,
) -> dict[str, list[str]]:
    """
    各月T-1時点で、その月のUniverse Cプール内トレイリング・コンポジットリターン上位fixed_n銘柄を選ぶ。
    ルックアヘッド防止: month_ts より前のデータのみ使用（calc_composite_return自体もshift済み）。
    """
    result: dict[str, list[str]] = {}
    for month_key in sorted(monthly_universe.keys()):
        month_ts = pd.Timestamp(month_key)
        members = monthly_universe[month_key]
        scores: dict[str, float] = {}
        for sym in members:
            comp = composite_scores.get(sym)
            if comp is None:
                continue
            hist = comp.loc[comp.index < month_ts]
            if hist.empty:
                continue
            val = hist.iloc[-1]
            if pd.notna(val):
                scores[sym] = float(val)
        top = sorted(scores.items(), key=lambda kv: -kv[1])[:fixed_n]
        result[month_key] = [s for s, _ in top]
    return result


def compute_membership_turnover(membership: dict[str, list[str]]) -> dict:
    """月次turnover（retained/added/removed）・平均turnover率・銘柄別継続月数を算出する。"""
    keys = sorted(membership.keys())
    monthly_stats = []
    prev_set: set[str] | None = None
    for key in keys:
        curr_set = set(membership[key])
        if prev_set is not None and prev_set:
            retained = curr_set & prev_set
            added = curr_set - prev_set
            removed = prev_set - curr_set
            monthly_stats.append({
                "month": key, "retained": len(retained), "added": len(added), "removed": len(removed),
                "turnover_pct": round(100 * len(added) / max(1, len(prev_set)), 2),
            })
        prev_set = curr_set

    avg_turnover_pct = float(np.mean([m["turnover_pct"] for m in monthly_stats])) if monthly_stats else 0.0

    duration_counter: Counter[str] = Counter()
    for key in keys:
        for sym in membership[key]:
            duration_counter[sym] += 1
    durations = list(duration_counter.values())

    return {
        "monthly_turnover": monthly_stats,
        "avg_monthly_turnover_pct": round(avg_turnover_pct, 2),
        "median_membership_duration_months": float(np.median(durations)) if durations else 0.0,
        "mean_membership_duration_months": float(np.mean(durations)) if durations else 0.0,
        "n_unique_symbols_ever_included": len(duration_counter),
        "n_months": len(keys),
    }


# ────────────────────────────────────────────────────────────────────────── #
# 2) 月次ローリングRSR（各月の42名部分プール内でのみパーセンタイル計算）
# ────────────────────────────────────────────────────────────────────────── #
def build_monthly_rolling_rsr(
    dynamic_membership: dict[str, list[str]],
    composite_scores: dict[str, pd.Series],
) -> pd.DataFrame:
    """月ごとにその月の42名部分プール内でパーセンタイルランクを計算し、時系列で結合する。"""
    monthly_keys = sorted(dynamic_membership.keys())
    blocks = []
    for i, key in enumerate(monthly_keys):
        members = dynamic_membership[key]
        comp_cols = {s: composite_scores[s] for s in members if s in composite_scores}
        if not comp_cols:
            continue
        comp_df = pd.DataFrame(comp_cols)
        start_ts = pd.Timestamp(key)
        end_ts = pd.Timestamp(monthly_keys[i + 1]) if i + 1 < len(monthly_keys) else comp_df.index.max() + pd.Timedelta(days=1)
        window = comp_df.loc[(comp_df.index >= start_ts) & (comp_df.index < end_ts)]
        if window.empty:
            continue
        rsr_window = (window.rank(axis=1, pct=True) * 100).clip(0, 100)
        blocks.append(rsr_window)
    if not blocks:
        return pd.DataFrame()
    combined = pd.concat(blocks, axis=0).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined


# ────────────────────────────────────────────────────────────────────────── #
# 3) 月次dyn_rsr42_bear_rs0（Top30/Bear20をその月の42名限定all_symsで再適用）
# ────────────────────────────────────────────────────────────────────────── #
def build_dynamic_active_matrix(
    dynamic_membership: dict[str, list[str]],
    universe_raw: dict,
    topix_close: pd.Series,
    composite_scores: dict[str, pd.Series],
    sectors: dict[str, str],
    bear_exclude_sectors: list[str] | None,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    build_dyn_rsr42_active（無改変・既存エンジン関数）を月ごとに「その月の42名のみ」を
    all_symsとして呼び出し、結果を月単位で切り出して結合する（架構変更なし・呼び出し方法のみ月次化）。
    """
    monthly_keys = sorted(dynamic_membership.keys())
    period_start, period_end = pd.Timestamp(start), pd.Timestamp(end)
    blocks = []
    for i, key in enumerate(monthly_keys):
        month_start_ts = pd.Timestamp(key)
        month_end_ts = (
            pd.Timestamp(monthly_keys[i + 1]) if i + 1 < len(monthly_keys) else period_end + pd.Timedelta(days=1)
        )
        sub_start_ts = max(month_start_ts, period_start)
        sub_end_ts = min(month_end_ts - pd.Timedelta(days=1), period_end)
        if sub_start_ts > sub_end_ts:
            continue

        members = [s for s in dynamic_membership[key] if s in universe_raw]
        if len(members) < 3:
            continue
        comp_cols = {s: composite_scores[s] for s in members if s in composite_scores}
        if not comp_cols:
            continue
        month_rsr = (pd.DataFrame(comp_cols).rank(axis=1, pct=True) * 100).clip(0, 100)

        active = build_dyn_rsr42_active(
            universe_raw=universe_raw, topix_close=topix_close, rsr_df=month_rsr,
            all_syms=members, start=sub_start_ts.strftime("%Y-%m-%d"), end=sub_end_ts.strftime("%Y-%m-%d"),
            bear_exclude_sectors=bear_exclude_sectors,
            sym_sector_map=sectors if bear_exclude_sectors else None,
        )
        blocks.append(active)

    if not blocks:
        return pd.DataFrame()
    combined = pd.concat(blocks, axis=0).sort_index()
    combined = combined.groupby(combined.index).last().fillna(0)
    return combined


# ────────────────────────────────────────────────────────────────────────── #
# 4) 実セクター読込（database/market/master/companies.parquet・現在分類の遡及適用）
# ────────────────────────────────────────────────────────────────────────── #
def load_sector_map(codes: list[str]) -> dict[str, str]:
    companies = pd.read_parquet(DATABASE_MASTER_DIR / "companies.parquet", columns=["Code", "Sector33CodeName"])
    sector_map = companies.dropna(subset=["Sector33CodeName"]).set_index("Code")["Sector33CodeName"]
    sector_map = sector_map[sector_map != ""]
    return {c: str(sector_map.get(c, "不明")) for c in codes}


# ────────────────────────────────────────────────────────────────────────── #
# 5) 容量診断サマライザ（run_bt()の戻り値から抽出。新規計測ロジックなし）
# ────────────────────────────────────────────────────────────────────────── #
def capacity_diagnostics(raw: dict) -> dict:
    skip_detail = raw.get("_skip_detail", []) or []
    reason_counts: Counter[str] = Counter(d.get("skip_reason", d.get("reason", "UNKNOWN")) for d in skip_detail)
    return {
        "avg_candidates": raw.get("avg_candidates"),
        "avg_simultaneous_holdings": raw.get("avg_simultaneous_holdings"),
        "cap_saturation_rate_pct": raw.get("cap_saturation_rate_pct"),
        "days_at_max_positions": raw.get("days_at_max_positions"),
        "avg_idle_cash_ratio_pct": raw.get("avg_idle_cash_ratio_pct"),
        "dyn_universe_excluded_count": raw.get("dyn_universe_excluded_count"),
        "skip_stats": raw.get("skip_stats"),
        "missed_by_cap_count": raw.get("missed_by_cap_count"),
        "rejected_by_lot_count": raw.get("rejected_by_lot_count"),
        "admitted_by_ratio_count": raw.get("admitted_by_ratio_count"),
        "skip_reason_breakdown": dict(reason_counts),
    }


# ────────────────────────────────────────────────────────────────────────── #
# main
# ────────────────────────────────────────────────────────────────────────── #
def main() -> int:
    started = datetime.now(_JST)
    monthly_universe = load_universe_c()
    # IS+OOS窓に関係する月のみに限定（Study75Bと同様の扱い）
    monthly_universe = {k: v for k, v in monthly_universe.items() if v}

    print("[1/6] 全銘柄のcomposite returnを一括計算中（月次選抜・月次RSRで共有）...")
    uc_union = sorted({s for m in monthly_universe.values() for s in m})
    cfg_probe = load_strategy_config()
    min_bars = 252 + cfg_probe.fujiko.mom_period + 2
    panel, missing, short_history = load_panel(uc_union, min_bars)
    composite_scores = {sym: calc_composite_return(df["Close"]) for sym, df in panel.items()}
    print(f"  union={len(uc_union)} loaded={len(panel)} missing={len(missing)} short_history={len(short_history)}")

    print("[2/6] Dynamic RSR42（月次固定42名）を構築中...")
    dynamic_membership = build_dynamic_rsr42_membership(monthly_universe, composite_scores, FIXED_N)
    turnover_stats = compute_membership_turnover(dynamic_membership)
    membership_sizes = [len(v) for v in dynamic_membership.values()]
    print(f"  months={len(dynamic_membership)} size_min={min(membership_sizes)} "
          f"size_max={max(membership_sizes)} avg_turnover%={turnover_stats['avg_monthly_turnover_pct']}")

    membership_path = RESULTS_DIR / f"dynamic_rsr42_membership_{datetime.now(_JST).strftime('%Y-%m-%d')}.json"
    membership_path.write_text(json.dumps(dynamic_membership, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Saved: {membership_path}")

    print("[3/6] 実セクターを読込中（database/market/master/companies.parquet）...")
    dynamic_union = sorted({s for v in dynamic_membership.values() for s in v})
    sectors = load_sector_map(dynamic_union)

    print("[4/6] シナリオデータセットを構築中（build_scenario_dataset・エンジン無改変）...")
    ds = build_scenario_dataset(sectors)
    print(f"  loaded={len(ds['trade_syms'])} missing={len(ds['missing'])} short_history={len(ds['short_history'])}")

    print("[5/6] 月次ローリングRSR・月次dyn_rsr42_bear_rs0アクティブ行列を構築中...")
    rolling_rsr = build_monthly_rolling_rsr(dynamic_membership, composite_scores)
    ds["rsr_df"] = rolling_rsr  # 本番の静的rsr_dfを月次ロール版で置換（唯一の非標準ロジック）

    bc = ds["base_cfg"].risk_controls.bear_universe_filter
    bear_exclude = list(bc.excluded_sectors) if bc.enabled else None
    dynamic_active = build_dynamic_active_matrix(
        dynamic_membership, ds["universe_raw"], ds["topix_close"], composite_scores, sectors,
        bear_exclude, FULL_START, FULL_END,
    )

    print("[6/6] IS/OOS/WF5foldをRunB（Dynamic RSR42）でfresh run実行中...")
    windows: dict[str, dict] = {}
    for wname, s, e, years in (("IS", IS_START, IS_END, 7.0), ("OOS", OOS_START, OOS_END, 1.0)):
        dsw = window_dataset(ds, s, e)
        act = and_masks(dynamic_active, dsw["alive_df"])
        raw = run_bt(dsw, act, s, e)
        windows[wname] = {"summary": summarize(raw, years), "capacity": capacity_diagnostics(raw)}
        sm = windows[wname]["summary"]
        print(f"  [{wname}] CAGR={sm.get('cagr')} Trades={sm.get('n_trades')} "
              f"AvgSimulHoldings={sm.get('avg_simultaneous_holdings')} Exposure={sm.get('avg_exposure')}")

    wf_results = []
    for fold in WF_SEGS:
        dsw = window_dataset(ds, fold["oos_s"], fold["oos_e"])
        act = and_masks(dynamic_active, dsw["alive_df"])
        raw = run_bt(dsw, act, fold["oos_s"], fold["oos_e"])
        sm = summarize(raw, 1.0)
        wf_pass = (sm.get("cagr") or 0.0) > 0
        wf_results.append({"year": fold["year"], **sm, "wf_pass": wf_pass})
        print(f"  [WF {fold['year']}] CAGR={sm.get('cagr')} {'PASS' if wf_pass else 'FAIL'}")

    wf_cagrs = [w["cagr"] for w in wf_results if w.get("cagr") is not None]
    wf_summary = {
        "segments": wf_results,
        "pass_count": sum(1 for w in wf_results if w["wf_pass"]),
        "avg_cagr": round(float(np.mean(wf_cagrs)), 2) if wf_cagrs else None,
    }

    delta_dynamic_is = round((windows["IS"]["summary"].get("cagr") or 0.0) - U0_REFERENCE["IS"]["cagr"], 2)
    delta_dynamic_oos = round((windows["OOS"]["summary"].get("cagr") or 0.0) - U0_REFERENCE["OOS"]["cagr"], 2)

    out = {
        "study": "Study76_prerequisite_D_ATR_EQ_universe_c_rebaseline",
        "canon_note": (
            "reports/study76_execution_plan.mdが定義する「Study76」(Clenow純正ベンチマーク)とは別物。"
            "本結果はStudy76が比較対象として必要とする前提工程（D_ATR_EQのStudy75 Universe再測定）。"
        ),
        "generated_at": datetime.now(_JST).isoformat(),
        "started_at": started.isoformat(),
        "config": {
            "fixed_n": FIXED_N, "capital": CAPITAL, "min_hold": MIN_HOLD,
            "is_window": [IS_START, IS_END], "oos_window": [OOS_START, OOS_END],
            "sector_source": "database/market/master/companies.parquet (current classification, retroactively applied)",
        },
        "primary_metric_delta_dynamic": {
            "definition": "RunB(Dynamic RSR42) - U0(static hindsight RSR42)",
            "IS_pp": delta_dynamic_is, "OOS_pp": delta_dynamic_oos,
        },
        "run_b_dynamic_rsr42": {
            "IS": windows["IS"]["summary"], "OOS": windows["OOS"]["summary"],
            "IS_capacity": windows["IS"]["capacity"], "OOS_capacity": windows["OOS"]["capacity"],
            "wf_5fold": wf_summary,
        },
        "u0_reference_static_rsr42": U0_REFERENCE,
        "official_production_reference": OFFICIAL_PRODUCTION_REFERENCE,
        "runA_negative_control_reference": NEGATIVE_CONTROL_RUNA_REFERENCE,
        "membership_turnover": turnover_stats,
        "membership_file": str(membership_path),
        "data_quality": {"union_symbols": len(uc_union), "loaded": len(panel),
                          "missing": len(missing), "short_history_excluded": len(short_history)},
    }

    out_path = RESULTS_DIR / f"study76_datr_eq_universe_c_rebaseline_{datetime.now(_JST).strftime('%Y-%m-%d')}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved: {out_path}")
    print(f"\nΔ_dynamic (RunB - U0): IS={delta_dynamic_is:+.2f}pp OOS={delta_dynamic_oos:+.2f}pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
