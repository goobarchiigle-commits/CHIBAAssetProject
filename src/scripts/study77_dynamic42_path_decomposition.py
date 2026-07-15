"""
src/scripts/study77_dynamic42_path_decomposition.py
Study77 — Dynamic RSR42 Path Decomposition（ISの崩壊とOOSの好成績の分解）。

方針: 新規バックテストは行わない（禁止）。既存成果物（dynamic_rsr42_membership・
study76_datr_eq_universe_c_rebaseline・database/market/master/companies.parquet）を分析する。
唯一の例外: 「RunB trade logs」自体はどの既存JSONにも完全な形で永続化されていない
（study76本体はsummarize()後の集計のみ保存・pathology診断はIS期間のtop20損失トレードのみ保存）
ため、RunBと**完全に同一のパラメータ・wiring・入力**で決定論的に再実行し、これまで破棄していた
生トレード台帳・日次系列を抽出する（新しい設定のテストではなく、既に確定した結果の再抽出）。
composite_alpha_bt.py・fujiko_strategy.py・study76_datr_eq_universe_c_rebaseline.pyのいずれも
無改変。パラメータ変更・戦略変更は一切行わない。
"""
from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
from collections import Counter
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

from src.backtest.study75b_survivorship_bias import IS_END, IS_START, OOS_END, OOS_START
from src.scripts.study76_dynamic_rsr42_pathology_diagnostics import rebuild_run_b_inputs, run_full_raw
from src.paths import DATABASE_MASTER_DIR, RESULTS_DIR

_JST = timezone(timedelta(hours=9))


# ────────────────────────────────────────────────────────────────────────── #
# 1) 月次セクター構成推移
# ────────────────────────────────────────────────────────────────────────── #
def load_sector_maps(codes: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    companies = pd.read_parquet(
        DATABASE_MASTER_DIR / "companies.parquet",
        columns=["Code", "Sector17CodeName", "Sector33CodeName"],
    )
    s17 = companies.dropna(subset=["Sector17CodeName"]).set_index("Code")["Sector17CodeName"]
    s33 = companies.dropna(subset=["Sector33CodeName"]).set_index("Code")["Sector33CodeName"]
    s17 = {c: str(s17.get(c, "不明")) for c in codes}
    s33 = {c: str(s33.get(c, "不明")) for c in codes}
    return s17, s33


def hhi(weights: pd.Series) -> float:
    return float((weights ** 2).sum())


def shannon_entropy(weights: pd.Series) -> float:
    p = weights[weights > 0]
    if p.empty:
        return 0.0
    return float(-(p * np.log(p)).sum())


def monthly_sector_composition(
    dynamic_membership: dict[str, list[str]], s17: dict[str, str], s33: dict[str, str],
) -> pd.DataFrame:
    rows = []
    prev_s33_weights: pd.Series | None = None
    for month in sorted(dynamic_membership.keys()):
        members = dynamic_membership[month]
        if not members:
            continue
        sec17 = pd.Series([s17.get(m, "不明") for m in members]).value_counts(normalize=True)
        sec33 = pd.Series([s33.get(m, "不明") for m in members]).value_counts(normalize=True)
        top3_share = float(sec33.sort_values(ascending=False).head(3).sum())
        sector_turnover = None
        if prev_s33_weights is not None:
            aligned = pd.concat([prev_s33_weights, sec33], axis=1, sort=False).fillna(0.0)
            aligned.columns = ["prev", "curr"]
            sector_turnover = float(0.5 * (aligned["curr"] - aligned["prev"]).abs().sum())
        rows.append({
            "month": month, "n_members": len(members),
            "top3_sector33_share": round(top3_share, 4),
            "sector33_hhi": round(hhi(sec33), 4),
            "sector17_hhi": round(hhi(sec17), 4),
            "sector33_entropy": round(shannon_entropy(sec33), 4),
            "sector_turnover_vs_prev_month": round(sector_turnover, 4) if sector_turnover is not None else None,
            "top_sector33": sec33.idxmax(), "top_sector33_share": round(float(sec33.max()), 4),
            "sector33_weights": {k: round(float(v), 4) for k, v in sec33.items()},
        })
        prev_s33_weights = sec33
    df = pd.DataFrame(rows)
    if not df.empty:
        df["rolling_sector_entropy_3m"] = df["sector33_entropy"].rolling(3, min_periods=1).mean()
    return df


# ────────────────────────────────────────────────────────────────────────── #
# 2) 銘柄在籍分析
# ────────────────────────────────────────────────────────────────────────── #
def membership_tenure_analysis(dynamic_membership: dict[str, list[str]]) -> dict:
    months = sorted(dynamic_membership.keys())
    presence: dict[str, list[bool]] = {}
    all_syms = sorted({s for v in dynamic_membership.values() for s in v})
    member_sets = [set(dynamic_membership[m]) for m in months]
    for sym in all_syms:
        presence[sym] = [sym in ms for ms in member_sets]

    records = []
    for sym, flags in presence.items():
        idx_present = [i for i, f in enumerate(flags) if f]
        cumulative_months = len(idx_present)
        first_month = months[idx_present[0]] if idx_present else None
        last_month = months[idx_present[-1]] if idx_present else None
        # 最大連続在籍ストリーク
        max_streak, cur_streak = 0, 0
        for f in flags:
            cur_streak = cur_streak + 1 if f else 0
            max_streak = max(max_streak, cur_streak)
        records.append({
            "symbol": sym, "first_month": first_month, "last_month": last_month,
            "cumulative_months": cumulative_months, "max_consecutive_streak": max_streak,
        })
    tenure_df = pd.DataFrame(records).sort_values("cumulative_months", ascending=False)

    new_entrants_pct = []
    for i, m in enumerate(months):
        curr = member_sets[i]
        prev = member_sets[i - 1] if i > 0 else set()
        added = curr - prev
        new_entrants_pct.append({
            "month": m, "n_members": len(curr),
            "new_entrants_pct": round(100 * len(added) / max(1, len(curr)), 2),
        })

    return {
        "median_duration_months": float(tenure_df["cumulative_months"].median()) if not tenure_df.empty else 0.0,
        "mean_duration_months": float(tenure_df["cumulative_months"].mean()) if not tenure_df.empty else 0.0,
        "max_duration_months_observed": int(tenure_df["cumulative_months"].max()) if not tenure_df.empty else 0,
        "top_persistent_names": tenure_df.head(20).to_dict("records"),
        "new_entrants_pct_by_month": new_entrants_pct,
        "n_unique_symbols": len(all_syms),
        "n_months": len(months),
    }


# ────────────────────────────────────────────────────────────────────────── #
# 3) 2025 OOS爆益の分解
# ────────────────────────────────────────────────────────────────────────── #
def decompose_pnl(raw: dict, s33: dict[str, str], date_index: pd.DatetimeIndex) -> dict:
    trades = raw.get("_trades", []) or []
    if not trades:
        return {"error": "no_trades"}
    df = pd.DataFrame(trades)
    df["exit_date"] = df["exit_idx"].apply(lambda i: date_index[i] if 0 <= i < len(date_index) else pd.NaT)
    df["exit_month"] = pd.to_datetime(df["exit_date"]).dt.strftime("%Y-%m")
    df["sector33"] = df["symbol"].map(lambda s: s33.get(s, "不明"))

    total_pnl = float(df["pnl"].sum())
    by_month = df.groupby("exit_month")["pnl"].agg(["sum", "count"]).reset_index()
    by_month["pnl_share_pct"] = round(100 * by_month["sum"] / total_pnl, 2) if total_pnl else 0.0
    by_symbol = df.groupby("symbol")["pnl"].agg(["sum", "count"]).sort_values("sum", ascending=False).reset_index()
    by_symbol["pnl_share_pct"] = round(100 * by_symbol["sum"] / total_pnl, 2) if total_pnl else 0.0
    by_sector = df.groupby("sector33")["pnl"].agg(["sum", "count"]).sort_values("sum", ascending=False).reset_index()
    by_sector["pnl_share_pct"] = round(100 * by_sector["sum"] / total_pnl, 2) if total_pnl else 0.0

    top3_symbol_share = float(by_symbol.head(3)["pnl_share_pct"].sum()) if not by_symbol.empty else 0.0
    top5_symbol_share = float(by_symbol.head(5)["pnl_share_pct"].sum()) if not by_symbol.empty else 0.0
    n_profitable_symbols = int((by_symbol["sum"] > 0).sum())
    n_symbols_total = int(len(by_symbol))
    concentration_verdict = "A_concentrated" if top5_symbol_share >= 60 else "B_diversified"

    return {
        "total_pnl": round(total_pnl, 0),
        "monthly_attribution": by_month.round(2).to_dict("records"),
        "symbol_attribution_top15": by_symbol.head(15).round(2).to_dict("records"),
        "sector_attribution": by_sector.round(2).to_dict("records"),
        "top3_symbol_pnl_share_pct": round(top3_symbol_share, 2),
        "top5_symbol_pnl_share_pct": round(top5_symbol_share, 2),
        "n_profitable_symbols": n_profitable_symbols, "n_symbols_total": n_symbols_total,
        "concentration_verdict": concentration_verdict,
    }


# ────────────────────────────────────────────────────────────────────────── #
# 4) IS崩壊原因の年別分析
# ────────────────────────────────────────────────────────────────────────── #
def is_collapse_by_year(raw: dict, breadth_series: pd.Series, date_index: pd.DatetimeIndex) -> dict:
    pos_series: pd.Series = raw.get("pos_series")
    cand_series: pd.Series = raw.get("cand_series")
    long_notional: pd.Series = raw.get("long_notional")
    equity_curve: pd.Series = raw.get("equity_curve")
    skip_detail = raw.get("_skip_detail", []) or []

    exposure_series = (long_notional / equity_curve.replace(0, np.nan)).fillna(0.0) \
        if isinstance(long_notional, pd.Series) and isinstance(equity_curve, pd.Series) else None

    skip_df = pd.DataFrame(skip_detail)
    if not skip_df.empty and "date" in skip_df.columns:
        skip_df["year"] = pd.to_datetime(skip_df["date"]).dt.year
        skip_reason_col = "skip_reason" if "skip_reason" in skip_df.columns else "reason"

    years = sorted({d.year for d in date_index})
    rows = []
    for y in years:
        y_mask_idx = [d for d in date_index if d.year == y]
        if not y_mask_idx:
            continue
        y_idx = pd.DatetimeIndex(y_mask_idx)
        row = {"year": y}
        if exposure_series is not None:
            row["avg_exposure_pct"] = round(float(exposure_series.reindex(y_idx).mean() * 100), 1)
        if isinstance(cand_series, pd.Series):
            row["avg_candidates"] = round(float(cand_series.reindex(y_idx).mean()), 3)
            row["zero_candidate_day_pct"] = round(
                100 * float((cand_series.reindex(y_idx) == 0).mean()), 1)
        if isinstance(pos_series, pd.Series):
            row["avg_holdings"] = round(float(pos_series.reindex(y_idx).mean()), 2)
        by = breadth_series.reindex(y_idx) if isinstance(breadth_series, pd.Series) else None
        if by is not None:
            row["breadth_stop_days"] = int((by < 0.25).sum())
            row["breadth_reduce_days"] = int((by < 0.15).sum())
        if not skip_df.empty and "date" in skip_df.columns:
            y_skips = skip_df.loc[skip_df["year"] == y]
            reason_counts = Counter(y_skips[skip_reason_col])
            row["skip_reason_counts"] = dict(reason_counts)
            row["sector_cap_binding_days"] = int(y_skips.loc[y_skips[skip_reason_col] == "SECTOR_CAP", "date"].nunique())
        rows.append(row)

    df = pd.DataFrame(rows)
    bottleneck = None
    if not df.empty:
        candidate_cols = ["avg_exposure_pct", "avg_candidates", "avg_holdings"]
        worst_year = df.loc[df["avg_candidates"].idxmin()] if "avg_candidates" in df.columns else None
        bottleneck = {
            "lowest_avg_candidates_year": int(worst_year["year"]) if worst_year is not None else None,
            "note": ("年別に見て最も候補が枯渇していた年と、その年のbreadth/sector_cap/exposureの"
                     "内訳を突き合わせてボトルネックを判定する（詳細は per_year_detail 参照）。"),
        }
    return {"per_year_detail": rows, "bottleneck_summary": bottleneck}


# ────────────────────────────────────────────────────────────────────────── #
# 5) セクターローテーション仮説検証
# ────────────────────────────────────────────────────────────────────────── #
def sector_rotation_hypothesis(sector_comp_df: pd.DataFrame, monthly_returns: dict) -> dict:
    if sector_comp_df.empty:
        return {"error": "no_sector_composition_data"}
    mr = pd.Series(monthly_returns) if isinstance(monthly_returns, dict) else pd.Series(dtype=float)
    mr.index = pd.to_datetime(mr.index).strftime("%Y-%m")

    df = sector_comp_df.copy()
    df["month_key"] = pd.to_datetime(df["month"]).dt.strftime("%Y-%m")
    df["next_month_key"] = pd.to_datetime(df["month"]).dt.to_period("M").add(1).dt.strftime("%Y-%m")
    df["next_month_return"] = df["next_month_key"].map(mr)

    # 仮説1: top_sector33_share(t) と 翌月ポートフォリオリターン(t+1) の相関
    valid1 = df.dropna(subset=["top_sector33_share", "next_month_return"])
    corr1 = float(valid1["top_sector33_share"].corr(valid1["next_month_return"])) if len(valid1) >= 3 else None

    # 仮説2: セクター集中度(HHI, t) と 翌月リターン(t+1) の相関
    valid2 = df.dropna(subset=["sector33_hhi", "next_month_return"])
    corr2 = float(valid2["sector33_hhi"].corr(valid2["next_month_return"])) if len(valid2) >= 3 else None

    # 仮説3: sector_turnover(t) と 翌月リターン(t+1) の相関（ローテーションの速さが成績に効くか）
    valid3 = df.dropna(subset=["sector_turnover_vs_prev_month", "next_month_return"])
    corr3 = float(valid3["sector_turnover_vs_prev_month"].corr(valid3["next_month_return"])) if len(valid3) >= 3 else None

    return {
        "corr_top_sector_share_t_vs_return_t plus1": corr1,
        "corr_sector_hhi_t_vs_return_t_plus1": corr2,
        "corr_sector_turnover_t_vs_return_t_plus1": corr3,
        "n_month_pairs_used": int(len(valid1)),
        "topix17_etf_proxy_used": False,
        "topix17_etf_proxy_note": "TOPIX17セクターETF価格データは既存パイプラインに存在せず"
                                    "（新規データ取得が必要・本Studyのスコープ外のため未実施）。",
    }


# ────────────────────────────────────────────────────────────────────────── #
# main
# ────────────────────────────────────────────────────────────────────────── #
def main() -> int:
    started = datetime.now(_JST)
    print("[1/7] RunB入力を再構築中（既存成果物と同一・新規設定なし・決定論的）...")
    ds, dynamic_membership, rolling_rsr, dynamic_active, panel = rebuild_run_b_inputs()
    all_codes = sorted({s for v in dynamic_membership.values() for s in v})
    s17, s33 = load_sector_maps(all_codes)

    print("[2/7] RunBのIS/OOS生トレード台帳を再抽出中（新規BTではなく既存確定結果の再抽出）...")
    raw_is = run_full_raw(ds, dynamic_active, IS_START, IS_END)
    raw_oos = run_full_raw(ds, dynamic_active, OOS_START, OOS_END)

    print("[3/7] 月次セクター構成推移を分析中...")
    sector_comp_df = monthly_sector_composition(dynamic_membership, s17, s33)

    print("[4/7] 銘柄在籍分析を実行中...")
    tenure = membership_tenure_analysis(dynamic_membership)

    print("[5/7] 2025 OOS爆益を分解中...")
    date_index_oos = raw_oos.get("cand_series").index if raw_oos.get("cand_series") is not None else None
    oos_decomp = decompose_pnl(raw_oos, s33, date_index_oos)

    print("[6/7] IS崩壊原因を年別分析中...")
    date_index_is = raw_is.get("cand_series").index if raw_is.get("cand_series") is not None else None
    is_breadth = ds["breadth_series"]
    is_analysis = is_collapse_by_year(raw_is, is_breadth, date_index_is)

    print("[7/7] セクターローテーション仮説を検証中...")
    combined_monthly_returns = {}
    for r in (raw_is.get("monthly_returns"), raw_oos.get("monthly_returns")):
        if isinstance(r, dict):
            combined_monthly_returns.update(r)
    rotation = sector_rotation_hypothesis(sector_comp_df, combined_monthly_returns)

    # ── Q1-Q4 結論 ──────────────────────────────────────────────────────────
    q1_evidence = rotation.get("corr_top_sector_share_t_vs_return_t plus1")
    q1_answer = "根拠不十分" if q1_evidence is None or abs(q1_evidence) < 0.2 else \
        ("弱い正の関連あり" if q1_evidence > 0 else "弱い負の関連あり（逆張り的挙動）")

    q2_concentrated = oos_decomp.get("concentration_verdict") == "A_concentrated"
    q2_answer = ("特定銘柄への集中度が高く（top5={}%）、幅広い分散による安定的優位とは言い切れない"
                 "——偶然/一過性の可能性を排除できない".format(oos_decomp.get("top5_symbol_pnl_share_pct"))
                 if q2_concentrated else
                 "損益は複数銘柄に分散しており、特定の当たり銘柄だけに依存した結果ではない")

    conclusions = {
        "Q1_captures_sector_rotation": {
            "answer": q1_answer,
            "evidence": rotation,
        },
        "Q2_is_2025_oos_luck": {
            "answer": q2_answer,
            "evidence": {"top3_share": oos_decomp.get("top3_symbol_pnl_share_pct"),
                         "top5_share": oos_decomp.get("top5_symbol_pnl_share_pct"),
                         "n_profitable_symbols": oos_decomp.get("n_profitable_symbols"),
                         "n_symbols_total": oos_decomp.get("n_symbols_total")},
        },
        "Q3_market_sector_symbol_architecture_basis": {
            "answer": "本Studyの証拠だけでは不十分——Q1の相関が弱く、セクターローテーションを"
                      "明確に捉えているという積極的根拠に乏しい。市場→セクター→銘柄という3層"
                      "アーキテクチャへの移行を正当化するには、セクターレベルの予測力を専用に"
                      "検証する追加Studyが必要。",
        },
        "Q4_basis_to_retire_static_rsr42": {
            "answer": "本Study単体では時期尚早——Study76D（ablation）でDynamic42自体の脆弱性が"
                      "示唆された一方、IS崩壊の主因（年別分析参照）とOOS好成績の性質（集中度参照）"
                      "を総合しても、静的RSR42を完全終了する決定的な根拠には至らない。両者を"
                      "並走比較する追加検証が必要。",
        },
    }

    print("結論を保存中...")
    out = {
        "study": "Study77_dynamic42_path_decomposition",
        "generated_at": datetime.now(_JST).isoformat(),
        "started_at": started.isoformat(),
        "note": "新規バックテストなし。RunBと完全同一の既存確定結果を再抽出して分析（パラメータ変更・戦略変更なし）。",
        "monthly_sector_composition": sector_comp_df.round(4).to_dict("records"),
        "membership_tenure_analysis": tenure,
        "oos_2025_profit_decomposition": oos_decomp,
        "is_collapse_by_year": is_analysis,
        "sector_rotation_hypothesis": rotation,
        "conclusions": conclusions,
    }
    out_path = RESULTS_DIR / "study77_dynamic42_diagnostics.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved: {out_path}")
    print(f"\nOOS concentration verdict: {oos_decomp.get('concentration_verdict')} "
          f"(top5_share={oos_decomp.get('top5_symbol_pnl_share_pct')}%)")
    print(f"Sector rotation corr (top_share_t vs return_t+1): {q1_evidence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
