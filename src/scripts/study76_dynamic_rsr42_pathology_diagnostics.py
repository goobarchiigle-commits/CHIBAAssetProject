"""
src/scripts/study76_dynamic_rsr42_pathology_diagnostics.py
Dynamic RSR42（study76_datr_eq_universe_c_rebaseline.py）病理診断 — 純粋診断専用。

パラメータ変更・最適化は一切行わない。RunBの構成を完全に同一のまま再実行し（決定論的・
乱数なし・同一結果になるはず）、summarize()で捨てられていた生の詳細（cand_series・_trades・
equity_curve等）を保持して分析する。

調査項目（ユーザー指定）:
  1. フィルタ段階別候補数（Universe42→RSR>=75→Top30→breadth→sector cap→position cap）
  2. 候補数の月次分布
  3. 年別exposure
  4. 損失最大トレード一覧
  5. rolling rsr_df と dynamic membership の日付整合性
  6. 1ヶ月ラグの取り違えチェック
  7. candidate_count==0の月数
"""
from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

from src.backtest.study75b_survivorship_bias import (
    IS_END, IS_START, build_scenario_dataset, window_dataset, and_masks, load_panel, load_universe_c,
)
import src.backtest.composite_alpha_bt as cab
from src.backtest.rsr import calc_composite_return
from src.backtest.study76_datr_eq_universe_c_rebaseline import (
    FIXED_N, FULL_START, FULL_END,
    build_dynamic_rsr42_membership, build_monthly_rolling_rsr, build_dynamic_active_matrix,
    load_sector_map,
)
from src.config_loader import load_strategy_config
from src.paths import RESULTS_DIR

_JST = timezone(timedelta(hours=9))


def rebuild_run_b_inputs():
    """study76_datr_eq_universe_c_rebaseline.py と完全同一の手順でRunBの入力を再構築する（決定論的）。"""
    monthly_universe = load_universe_c()
    monthly_universe = {k: v for k, v in monthly_universe.items() if v}
    uc_union = sorted({s for m in monthly_universe.values() for s in m})

    cfg_probe = load_strategy_config()
    min_bars = 252 + cfg_probe.fujiko.mom_period + 2
    panel, missing, short_history = load_panel(uc_union, min_bars)
    composite_scores = {sym: calc_composite_return(df["Close"]) for sym, df in panel.items()}

    dynamic_membership = build_dynamic_rsr42_membership(monthly_universe, composite_scores, FIXED_N)

    dynamic_union = sorted({s for v in dynamic_membership.values() for s in v})
    sectors = load_sector_map(dynamic_union)

    ds = build_scenario_dataset(sectors)
    rolling_rsr = build_monthly_rolling_rsr(dynamic_membership, composite_scores)
    ds["rsr_df"] = rolling_rsr

    bc = ds["base_cfg"].risk_controls.bear_universe_filter
    bear_exclude = list(bc.excluded_sectors) if bc.enabled else None
    dynamic_active = build_dynamic_active_matrix(
        dynamic_membership, ds["universe_raw"], ds["topix_close"], composite_scores, sectors,
        bear_exclude, FULL_START, FULL_END,
    )
    return ds, dynamic_membership, rolling_rsr, dynamic_active, panel


def run_full_raw(ds: dict, dynamic_active: pd.DataFrame, start: str, end: str) -> dict:
    """run_bt相当だが summarize() を経由せず raw dict をそのまま返す（診断用フル情報保持）。"""
    dsw = window_dataset(ds, start, end)
    act = and_masks(dynamic_active, dsw["alive_df"])
    return cab.run_scenario(
        scenario="BASELINE",
        universe_raw=dsw["universe_raw"], rsr_df=dsw["rsr_df"], alpha_df=None,
        regime_df=dsw["regime_df"], trade_syms=dsw["trade_syms"], rsr_syms=dsw["rsr_syms"],
        cfg=dsw["base_cfg"], start=start, end=end, verbose=False,
        tech_matrices=dsw["tech_matrices"], breadth_series=dsw["breadth_series"],
        capital=3_000_000, min_hold=3, topix_close=dsw["topix_close"],
        market_shock_mode="composite", rsr_exit_threshold=70.0,
        sym_active_df=act,
        enable_simple_rsr_exit=True, enable_atr_trailing_prod=True,
        enable_multilayer_rsr=True, enable_atr_risk_sizing=False,
        enable_mtf_filter=False, sizing_mode="existing",
        exit_policy="A", addon_policy="D", addon_size_frac=0.25, addon_atr_mult=1.0,
    )


def diagnose_date_alignment(rolling_rsr: pd.DataFrame, dynamic_membership: dict, sample_n: int = 3) -> dict:
    """項目5・6: rolling_rsrのNaNパターンがdynamic_membershipと厳密に一致するか・月境界ズレがないかを確認する。"""
    results = []
    months = sorted(dynamic_membership.keys())
    # 複数月にまたがって出入りする銘柄をサンプルする(在籍期間が短い銘柄ほど検証価値が高い)
    from collections import Counter
    counts = Counter(s for v in dynamic_membership.values() for s in v)
    sample_syms = [s for s, c in sorted(counts.items(), key=lambda kv: kv[1])[:sample_n]]

    for sym in sample_syms:
        if sym not in rolling_rsr.columns:
            results.append({"symbol": sym, "error": "not_in_rolling_rsr_columns"})
            continue
        series = rolling_rsr[sym]
        member_months = {m for m in months if sym in dynamic_membership[m]}
        # 在籍月の日付範囲でNaNでないか、非在籍月の日付範囲でNaNかを月ごとに確認
        mismatches = []
        for i, m in enumerate(months):
            m_start = pd.Timestamp(m)
            m_end = pd.Timestamp(months[i + 1]) if i + 1 < len(months) else series.index.max() + pd.Timedelta(days=1)
            window = series.loc[(series.index >= m_start) & (series.index < m_end)]
            if window.empty:
                continue
            is_member = m in member_months
            all_valid = window.notna().all()
            all_nan = window.isna().all()
            if is_member and not all_valid:
                mismatches.append({"month": m, "expected": "all_valid(member)", "found": "has_nan"})
            if (not is_member) and not all_nan:
                mismatches.append({"month": m, "expected": "all_nan(non_member)", "found": "has_value"})
        results.append({
            "symbol": sym, "n_member_months": len(member_months), "n_total_months": len(months),
            "mismatches": mismatches, "mismatch_count": len(mismatches),
        })
    return {"sampled_symbols": sample_syms, "per_symbol": results}


def diagnose_ffill_contamination(ds: dict, dynamic_membership: dict, sample_n: int = 3) -> dict:
    """
    FujikoStrategyが実際に見るrsr_series(ds['rsr_df'][sym])に対しffill()した場合、
    非在籍月にどれだけ「古い在籍時点の値」が漏れ込むかを直接シミュレートして定量化する。
    （fujiko_strategy.py::_slice_series_to_array の fill_method="ffill" と同一操作）
    """
    rolling_rsr = ds["rsr_df"]
    months = sorted(dynamic_membership.keys())
    from collections import Counter
    counts = Counter(s for v in dynamic_membership.values() for s in v)
    sample_syms = [s for s, c in sorted(counts.items(), key=lambda kv: kv[1])[:sample_n]]

    findings = []
    for sym in sample_syms:
        if sym not in rolling_rsr.columns:
            continue
        raw = rolling_rsr[sym]
        ffilled = raw.ffill()
        leaked_mask = raw.isna() & ffilled.notna()
        leaked_days = int(leaked_mask.sum())
        leaked_value_sample = ffilled.loc[leaked_mask].head(5).to_dict()
        findings.append({
            "symbol": sym,
            "total_days": len(raw),
            "raw_valid_days": int(raw.notna().sum()),
            "days_leaked_by_ffill": leaked_days,
            "leaked_pct_of_total": round(100 * leaked_days / max(1, len(raw)), 1),
            "sample_leaked_stale_rsr_values": {str(k): round(float(v), 1) for k, v in leaked_value_sample.items()},
        })
    return {"note": "非在籍月にffill()を適用した場合に漏れ込む「古いRSR値」の日数を直接計測（シミュレーション）。"
                     "FujikoStrategy.precompute_signals()が実際にこの操作をrsr_seriesへ適用している"
                     "（fujiko_strategy.py L279 fill_method='ffill'・コード読解で確認済み）。",
            "per_symbol": findings}


def main() -> int:
    started = datetime.now(_JST)
    print("[1/5] RunB入力を再構築中（決定論的・完全同一手順）...")
    ds, dynamic_membership, rolling_rsr, dynamic_active, panel = rebuild_run_b_inputs()

    print("[2/5] IS期間でfull raw出力を取得中（summarize()を経由しない生データ保持）...")
    raw = run_full_raw(ds, dynamic_active, IS_START, IS_END)

    print("[3/5] 候補数の月次分布・ゼロ候補月を集計中...")
    cand_series: pd.Series = raw.get("cand_series")
    monthly_cand = cand_series.resample("MS").agg(["mean", "max", "min", lambda s: int((s == 0).sum())])
    monthly_cand.columns = ["mean", "max", "min", "zero_days"]
    zero_candidate_months = monthly_cand.loc[monthly_cand["mean"] == 0]
    n_zero_months = len(zero_candidate_months)
    n_total_months = len(monthly_cand)
    total_zero_days = int((cand_series == 0).sum())
    total_days = len(cand_series)

    print("[4/5] 年別exposure・損失最大トレードを集計中...")
    date_index = cand_series.index  # exit_idx/entry_idx → 実日付への変換に使う（common_dates相当）
    equity_curve = raw.get("equity_curve")
    trades = raw.get("_trades", []) or []
    # SELLレコードには"date"列がなく entry_idx/exit_idx のみ持つ（composite_alpha_bt.py実装確認済み）
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty and "exit_idx" in trades_df.columns:
        trades_df["exit_date"] = trades_df["exit_idx"].apply(
            lambda idx: str(date_index[idx].date()) if 0 <= idx < len(date_index) else None)
        trades_df["entry_date"] = trades_df["entry_idx"].apply(
            lambda idx: str(date_index[idx].date()) if 0 <= idx < len(date_index) else None)

    pnl_by_year = {}
    if not trades_df.empty and "exit_date" in trades_df.columns:
        trades_df["year"] = pd.to_datetime(trades_df["exit_date"]).dt.year
        pnl_by_year = trades_df.groupby("year")["pnl"].agg(["sum", "count", "mean"]).to_dict("index")

    top_losers = []
    if not trades_df.empty and "pnl" in trades_df.columns:
        cols = [c for c in ("entry_date", "exit_date", "symbol", "entry", "exit", "qty", "pnl", "reason")
                if c in trades_df.columns]
        worst = trades_df.nsmallest(20, "pnl")
        top_losers = worst[cols].to_dict("records")

    # 損失最大トレードのエントリー月がdynamic_membershipに実在したかクロスチェック
    entry_month_check = []
    for t in top_losers[:10]:
        entry_date = pd.Timestamp(t.get("entry_date"))
        applicable_months = [m for m in sorted(dynamic_membership.keys()) if pd.Timestamp(m) <= entry_date]
        applicable_month = applicable_months[-1] if applicable_months else None
        sym = t.get("symbol")
        was_member = bool(applicable_month and sym in dynamic_membership.get(applicable_month, []))
        entry_month_check.append({
            "symbol": sym, "entry_date": t.get("entry_date"), "exit_date": t.get("exit_date"),
            "pnl": t.get("pnl"), "reason": t.get("reason"),
            "applicable_month": applicable_month, "was_dynamic_member_that_month": was_member,
        })

    print("[5/5] 日付整合性・ffill汚染・funnel診断を実行中...")
    alignment = diagnose_date_alignment(rolling_rsr, dynamic_membership)
    ffill_contamination = diagnose_ffill_contamination(ds, dynamic_membership)

    # funnel（測定可能な範囲のみ・測定不能段階は明記）
    sample_date = cand_series.index[len(cand_series) // 2]
    sample_month_key = max((m for m in dynamic_membership if pd.Timestamp(m) <= sample_date), default=None)
    funnel_sample = None
    if sample_month_key:
        members = dynamic_membership[sample_month_key]
        rsr_today = rolling_rsr.loc[sample_date, members].dropna() if sample_date in rolling_rsr.index else pd.Series(dtype=float)
        rsr_pass = int((rsr_today >= 75.0).sum())
        active_today = dynamic_active.loc[sample_date] if sample_date in dynamic_active.index else pd.Series(dtype=float)
        top_pass = int((active_today == 1).sum())
        funnel_sample = {
            "sample_date": str(sample_date.date()),
            "stage_1_universe42": len(members),
            "stage_2_rsr_geq_75_measured_from_clean_rolling_rsr": rsr_pass,
            "stage_3_top30_bear20_active_mask": top_pass,
            "stage_4_breadth": "day-level gate (risk_off flag), not per-symbol funnel — see breadth_series value separately",
            "stage_5_sector_cap": raw.get("skip_stats", {}).get("sector_cap"),
            "stage_6_position_cap": raw.get("missed_by_cap_count"),
            "caveat": (
                "stage_2はFujikoStrategy内部の実際のゲート判定(ffill汚染の影響を受ける可能性)ではなく、"
                "cleanなrolling_rsr(本来あるべき値)から独立に再計算した値。stage_5/6はIS全期間の累計値"
                "（sample_dateのみの値ではない・run_scenarioが日次分解を公開していないため）。"
            ),
        }

    out = {
        "study": "Study76_dynamic_rsr42_pathology_diagnostics",
        "generated_at": datetime.now(_JST).isoformat(),
        "started_at": started.isoformat(),
        "note": "パラメータ変更・最適化なし。RunBと完全同一構成の再実行（決定論的）+ 生データ保持による診断。",
        "candidate_count_distribution": {
            "avg_candidates_per_day": round(float(cand_series.mean()), 3),
            "total_days": total_days,
            "total_zero_candidate_days": total_zero_days,
            "zero_candidate_day_pct": round(100 * total_zero_days / max(1, total_days), 1),
            "n_months_total": n_total_months,
            "n_months_with_zero_mean_candidates": n_zero_months,
            "monthly_distribution_head20": monthly_cand.head(20).reset_index().rename(
                columns={"index": "month"}).astype(str).to_dict("records"),
        },
        "exposure": {"pnl_by_year": pnl_by_year, "note": "position-level daily exposure series not persisted by engine; "
                                                            "using trade-level pnl aggregation by exit year as proxy"},
        "top_20_losing_trades": top_losers,
        "top10_loser_dynamic_membership_crosscheck": entry_month_check,
        "date_alignment_check": alignment,
        "ffill_contamination_simulation": ffill_contamination,
        "funnel_sample": funnel_sample,
        "engine_level_counters_full_is_period": {
            "dyn_universe_excluded_count": raw.get("dyn_universe_excluded_count"),
            "skip_stats": raw.get("skip_stats"),
            "missed_by_cap_count": raw.get("missed_by_cap_count"),
            "rejected_by_lot_count": raw.get("rejected_by_lot_count"),
            "n_trades": raw.get("n_trades"), "max_dd": raw.get("max_dd"), "cagr": raw.get("cagr"),
        },
    }

    out_path = RESULTS_DIR / f"study76_dynamic_rsr42_pathology_diagnostics_{datetime.now(_JST).strftime('%Y-%m-%d')}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved: {out_path}")
    print(f"\navg_candidates={out['candidate_count_distribution']['avg_candidates_per_day']} "
          f"zero_candidate_days_pct={out['candidate_count_distribution']['zero_candidate_day_pct']}% "
          f"months_zero_mean={n_zero_months}/{n_total_months}")
    print(f"max_dd(re-verified)={raw.get('max_dd')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
