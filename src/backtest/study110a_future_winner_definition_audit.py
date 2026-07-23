"""
src/backtest/study110a_future_winner_definition_audit.py
Study110A — Future Winner Definition Audit（記述統計・Phase1・7ラベル版）

正典: ユーザータスク指示"Study110A Implementation"（2026-07-22・Phase1=7ラベルに縮小版）
      Study101（RSR42固定名簿=hindsight selection確定・FUJIKO 2.0凍結の起点）

目的（狭く固定）:
  「未来勝者」の定義を確定する。alpha探索ではない。Universe Generator設計の前段階として、
  異なるhorizon・異なるラベル定義（raw top-K% / Calmar調整top-K%）が同一の銘柄集合を
  指すのか、それとも別物なのかを記述統計のみで検証する。

禁止事項（厳守）: 新規alpha提案・最適化・新規データ取得・backtestは一切行わない。
  ラベル=記述目的の分類のみ（トレーディングルールではない）。

データ源（既存キャッシュのみ・Study95と同一）:
  - backtests/study75_rule_universe.json （Universe C・Study95/95Eと同一）
  - data/jquants/processed/{code}.parquet （価格・Study95と同一ソース・load_close_panel再利用）

Phase1ラベル（7種・27通りから縮小・スイープ禁止）:
  1. 3M Top10%   2. 3M Top5%
  3. 6M Top10%   4. 6M Top5%
  5. 12M Top10%  6. 12M Top5%
  7. 12M Calmar-adjusted Top10%（score = fwd_12M / abs(max_dd_12M)）

出力: winner overlap matrix（7x7 Jaccard・主判定=3M/6M/12M Top10の3x3）/
      persistence matrix（strict=同一ラベル残存率・loose=Top20%残存率）/
      concentration statistics（top1/2/5/10%が総プラスリターンに占める比率）/
      transition probabilities（persistenceと同一計算の別表現）
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

try:
    from src.paths import REPORTS_DIR, RESULTS_DIR
    import src.backtest.study95_cs_momentum_factor_level as s95
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.paths import REPORTS_DIR, RESULTS_DIR
    import src.backtest.study95_cs_momentum_factor_level as s95

RUN_DATE = "2026-07-22"
OUT_JSON = RESULTS_DIR / f"study110a_future_winner_definition_audit_{RUN_DATE}.json"
OUT_PANEL_CSV = RESULTS_DIR / f"study110a_panel_enriched_{RUN_DATE}.csv"
OUT_MD = REPORTS_DIR / "study110a_future_winner_definition_audit.md"

HORIZONS_MONTHS = {"3M": 3, "6M": 6, "12M": 12}
HORIZONS_DAYS = {"3M": 63, "6M": 126, "12M": 252}  # Study95と同一トレーディング日換算
MIN_CROSS_SECTION_N = 30  # Study95 assign_deciles と同型の最小サンプル閾値

LABELS = [
    ("3M_top10", "3M", "raw", 0.90),
    ("3M_top5", "3M", "raw", 0.95),
    ("6M_top10", "6M", "raw", 0.90),
    ("6M_top5", "6M", "raw", 0.95),
    ("12M_top10", "12M", "raw", 0.90),
    ("12M_top5", "12M", "raw", 0.95),
    ("12M_calmar_top10", "12M", "calmar", 0.90),
]
LOOSE_PCT = 0.80  # persistence "top20%残存" 判定用


# ---------------------------------------------------------------- forward return / max drawdown
def calc_fwd_return_row(close_df: pd.DataFrame, base_pos: int, horizon_days: int) -> pd.Series:
    n = len(close_df)
    pos_h = base_pos + horizon_days
    if pos_h >= n:
        return pd.Series(np.nan, index=close_df.columns)
    p0 = close_df.iloc[base_pos]
    p1 = close_df.iloc[pos_h]
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = (p1 / p0) - 1.0
    return ret.replace([np.inf, -np.inf], np.nan)


def calc_maxdd_row(close_df: pd.DataFrame, base_pos: int, horizon_days: int) -> pd.Series:
    """window=[base_pos, base_pos+horizon_days]でのrunning max drawdown（負値・0=無下落）。"""
    n = len(close_df)
    pos_h = base_pos + horizon_days
    if pos_h >= n:
        return pd.Series(np.nan, index=close_df.columns)
    window = close_df.iloc[base_pos: pos_h + 1]
    running_max = window.cummax()
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = window / running_max - 1.0
    return dd.min()  # 各銘柄の最大ドローダウン（<=0）


# ---------------------------------------------------------------- ラベル割当（月次クロスセクション内）
def assign_topk_labels(panel: pd.DataFrame) -> pd.DataFrame:
    for label_name, horizon, kind, pct_threshold in LABELS:
        value_col = f"fwd_{horizon}" if kind == "raw" else f"calmar_{horizon}"
        col = f"label_{label_name}"
        loose_col = f"labelloose_{label_name}"  # top20%版（persistence用）
        panel[col] = False
        panel[loose_col] = False
        for rb_date, idx in panel.groupby("rebalance_date").groups.items():
            sub = panel.loc[idx, value_col]
            valid = sub.dropna()
            if len(valid) < MIN_CROSS_SECTION_N:
                continue
            pct_rank = valid.rank(pct=True)
            panel.loc[valid.index[pct_rank >= pct_threshold], col] = True
            panel.loc[valid.index[pct_rank >= LOOSE_PCT], loose_col] = True
    return panel


# ---------------------------------------------------------------- ① Winner overlap matrix
def winner_overlap_matrix(panel: pd.DataFrame) -> dict:
    label_names = [l[0] for l in LABELS]
    jaccards: dict[str, dict[str, list[float]]] = {a: {b: [] for b in label_names} for a in label_names}
    for rb_date, g in panel.groupby("rebalance_date"):
        sets = {name: set(g.loc[g[f"label_{name}"], "code"]) for name in label_names}
        for a in label_names:
            for b in label_names:
                sa, sb = sets[a], sets[b]
                if not sa and not sb:
                    continue
                union = sa | sb
                if not union:
                    continue
                jaccards[a][b].append(len(sa & sb) / len(union))
    matrix = {a: {b: (round(float(np.mean(jaccards[a][b])), 4) if jaccards[a][b] else None)
                  for b in label_names} for a in label_names}

    # 主判定: 3M/6M/12M Top10 の3x3（decision treeの一次指標）
    primary = ["3M_top10", "6M_top10", "12M_top10"]
    primary_pairs = {}
    for i, a in enumerate(primary):
        for b in primary[i + 1:]:
            primary_pairs[f"{a}_vs_{b}"] = matrix[a][b]
    valid_vals = [v for v in primary_pairs.values() if v is not None]
    overall_mean = round(float(np.mean(valid_vals)), 4) if valid_vals else None

    def case_of(v: float | None) -> str | None:
        if v is None:
            return None
        if v > 0.50:
            return "Case1_single_ontology"
        if v >= 0.20:
            return "Case2_partial_multi_sleeve"
        return "Case3_multiple_ontology"

    return {
        "full_matrix_7x7": matrix,
        "primary_3m_6m_12m_top10_pairs": primary_pairs,
        "primary_mean_overlap": overall_mean,
        "primary_case": case_of(overall_mean),
        "case_boundaries": {"Case1_single_ontology": ">50%", "Case2_partial_multi_sleeve": "20-50%",
                            "Case3_multiple_ontology": "<20%"},
    }


# ---------------------------------------------------------------- ② Persistence / Transition
def persistence_and_transition(panel: pd.DataFrame, monthly_universe_keys: list[str]) -> dict:
    key_set = set(monthly_universe_keys)
    out = {}
    for label_name, horizon, kind, pct_threshold in LABELS:
        h_months = HORIZONS_MONTHS[horizon]
        strict_hits, strict_total = 0, 0
        loose_hits, loose_total = 0, 0
        transition_counts: dict[str, dict[str, int]] = {
            "winner": {"still_winner": 0, "top20_not_winner": 0, "outside_top20": 0, "no_data": 0}
        }
        panel_by_date = {rb: g for rb, g in panel.groupby("rebalance_date")}
        for rb_str in sorted(panel_by_date.keys()):
            t0 = pd.Timestamp(rb_str)
            t1_key = (t0 + pd.DateOffset(months=h_months)).strftime("%Y-%m-01")
            if t1_key not in key_set or t1_key not in panel_by_date:
                continue
            g0 = panel_by_date[rb_str]
            g1 = panel_by_date[t1_key]
            winners_t0 = set(g0.loc[g0[f"label_{label_name}"], "code"])
            if not winners_t0:
                continue
            still_winner_t1 = set(g1.loc[g1[f"label_{label_name}"], "code"])
            loose_t1 = set(g1.loc[g1[f"labelloose_{label_name}"], "code"])
            have_data_t1 = set(g1["code"])

            for code in winners_t0:
                strict_total += 1
                loose_total += 1
                if code in still_winner_t1:
                    strict_hits += 1
                    loose_hits += 1
                    transition_counts["winner"]["still_winner"] += 1
                elif code in loose_t1:
                    loose_hits += 1
                    transition_counts["winner"]["top20_not_winner"] += 1
                elif code in have_data_t1:
                    transition_counts["winner"]["outside_top20"] += 1
                else:
                    transition_counts["winner"]["no_data"] += 1

        out[label_name] = {
            "horizon_months": h_months,
            "strict_persistence_rate": round(strict_hits / strict_total, 4) if strict_total >= MIN_CROSS_SECTION_N else None,
            "loose_top20_persistence_rate": round(loose_hits / loose_total, 4) if loose_total >= MIN_CROSS_SECTION_N else None,
            "n_winner_observations": strict_total,
            "transition_counts": transition_counts["winner"],
        }
    return out


# ---------------------------------------------------------------- ③ Concentration statistics
def concentration_stats(panel: pd.DataFrame) -> dict:
    out = {}
    for horizon in ("3M", "6M", "12M"):
        value_col = f"fwd_{horizon}"
        shares = {"top1pct": [], "top2pct": [], "top5pct": [], "top10pct": []}
        for rb_date, g in panel.groupby("rebalance_date"):
            vals = g[value_col].dropna().sort_values(ascending=False)
            if len(vals) < MIN_CROSS_SECTION_N:
                continue
            positive_total = vals[vals > 0].sum()
            if positive_total <= 0:
                continue
            n = len(vals)
            for key, frac in (("top1pct", 0.01), ("top2pct", 0.02), ("top5pct", 0.05), ("top10pct", 0.10)):
                k = max(1, int(round(n * frac)))
                top_sum = vals.iloc[:k].clip(lower=0).sum()
                shares[key].append(float(top_sum / positive_total))
        out[horizon] = {k: (round(float(np.mean(v)), 4) if v else None) for k, v in shares.items()}
        out[horizon]["n_periods"] = len(shares["top10pct"])
    return out


def main() -> None:
    print("Study110A — Future Winner Definition Audit（記述統計のみ・Phase1=7ラベル・既存キャッシュのみ）")

    print("[1/5] Universe C / カレンダー読込（Study95と同一関数）...")
    monthly_universe = s95.load_universe()
    rebalance_dates = sorted(pd.Timestamp(k) for k in monthly_universe)
    topix = s95.load_topix_calendar()
    calendar = topix.index
    calendar_pos = {d: i for i, d in enumerate(calendar)}

    print(f"[2/5] 価格パネル読込（{len(rebalance_dates)}ヶ月・Study95と同一関数）...")
    all_codes = sorted({c for v in monthly_universe.values() for c in v})
    close_df, missing_codes = s95.load_close_panel(all_codes, calendar)
    print(f"  loaded={close_df.shape[1]}  missing={len(missing_codes)}")

    print("[3/5] 月次forward return / max drawdown / Calmar計算...")
    records: list[dict] = []
    for rb_date in rebalance_dates:
        rb_str = rb_date.strftime("%Y-%m-%d")
        universe_codes = monthly_universe[rb_str]
        if rb_date not in calendar_pos:
            later = calendar[calendar >= rb_date]
            if len(later) == 0:
                continue
            rb_date_eff = later[0]
        else:
            rb_date_eff = rb_date
        base_pos = calendar_pos[rb_date_eff]

        universe_mask = pd.Series(False, index=close_df.columns)
        universe_mask.loc[[c for c in universe_codes if c in close_df.columns]] = True

        fwd = {h: calc_fwd_return_row(close_df, base_pos, d).where(universe_mask) for h, d in HORIZONS_DAYS.items()}
        maxdd_12m = calc_maxdd_row(close_df, base_pos, HORIZONS_DAYS["12M"]).where(universe_mask)
        with np.errstate(divide="ignore", invalid="ignore"):
            calmar_12m = fwd["12M"] / maxdd_12m.abs()
        calmar_12m = calmar_12m.replace([np.inf, -np.inf], np.nan)

        rb_key = rb_date.strftime("%Y-%m-01")
        for code in universe_codes:
            if code not in close_df.columns:
                continue
            records.append({
                "rebalance_date": rb_key, "code": code,
                "fwd_3M": float(fwd["3M"].get(code, np.nan)),
                "fwd_6M": float(fwd["6M"].get(code, np.nan)),
                "fwd_12M": float(fwd["12M"].get(code, np.nan)),
                "maxdd_12M": float(maxdd_12m.get(code, np.nan)),
                "calmar_12M": float(calmar_12m.get(code, np.nan)),
            })

    panel = pd.DataFrame(records)
    print(f"  panel rows={len(panel):,}")

    print("[4/5] ラベル割当（月次クロスセクション内top-K%・7ラベル）...")
    panel = assign_topk_labels(panel)
    panel.to_csv(OUT_PANEL_CSV, index=False, encoding="utf-8")
    print(f"  Enriched panel CSV: {OUT_PANEL_CSV}")

    print("[5/5] Overlap matrix / Persistence / Concentration集計...")
    result: dict = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "n_panel_rows": int(len(panel)),
        "n_rebalance_dates": len(rebalance_dates),
        "labels": [l[0] for l in LABELS],
    }
    result["winner_overlap_matrix"] = winner_overlap_matrix(panel)
    result["persistence_and_transition"] = persistence_and_transition(panel, list(monthly_universe.keys()))
    result["concentration_statistics"] = concentration_stats(panel)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\nJSON: {OUT_JSON}")


if __name__ == "__main__":
    main()
