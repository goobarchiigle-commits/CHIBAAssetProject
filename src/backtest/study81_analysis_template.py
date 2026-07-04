"""
study81_analysis_template.py
Study81以降で「追加BTなしに」即実行できる統計解析テンプレート。

Study80Aで構築した観測基盤（trade_dataset_v2.json / missed_candidates_full.json /
forward_return_dataset.json / opportunity_cost_dataset.json / correlation_dataset.json）
のみを入力とする。新規BT・エンジン呼び出しは一切行わない。

提供する解析関数:
  1. compare_adopted_vs_missed()   採用 vs 見送り比較（Mann-Whitney U）
  2. forward_return_comparison()   Forward Return比較（KS検定）
  3. opportunity_cost_summary()    Opportunity Cost分析（既存datasetの整形出力）
  4. sector_concentration_test()   Sector集中分析（permutation test）
  5. portfolio_correlation_check() Portfolio相関分析（同日候補の相関観測の要約）
  6. time_diversification_test()   時間分散仮説（bootstrap CI）
  7. same_day_competition_test()   同日競合分析（rank0見送り率のbootstrap CI）

実行例: `python src/backtest/study81_analysis_template.py` で全解析を一括実行し
`reports/study81_analysis_output.md`相当のサマリをprintする（保存はStudy81側で選択）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[2]
BT_DIR = ROOT / "backtests"


def _load(name: str) -> dict:
    with open(BT_DIR / name, encoding="utf-8") as f:
        return json.load(f)


# ======================================================================
# 1. 採用 vs 見送り比較（Mann-Whitney U）
# ======================================================================
def compare_adopted_vs_missed(metric_adopted: list[float], metric_missed: list[float], label: str = "") -> dict:
    """Mann-Whitney U検定（分布の中心傾向差を検定・正規性を仮定しない）"""
    a = np.array([v for v in metric_adopted if v is not None], dtype=float)
    b = np.array([v for v in metric_missed if v is not None], dtype=float)
    if len(a) < 2 or len(b) < 2:
        return {"label": label, "error": "サンプル不足"}
    u_stat, p_val = sps.mannwhitneyu(a, b, alternative="two-sided")
    return {
        "label": label, "n_adopted": len(a), "n_missed": len(b),
        "median_adopted": round(float(np.median(a)), 3), "median_missed": round(float(np.median(b)), 3),
        "u_statistic": round(float(u_stat), 2), "p_value": round(float(p_val), 4),
        "significant_at_5pct": bool(p_val < 0.05),
    }


# ======================================================================
# 2. Forward Return比較（Kolmogorov-Smirnov検定）
# ======================================================================
def forward_return_comparison(group_a: list[float], group_b: list[float], label: str = "") -> dict:
    """KS検定（分布形状全体の差を検定）"""
    a = np.array([v for v in group_a if v is not None], dtype=float)
    b = np.array([v for v in group_b if v is not None], dtype=float)
    if len(a) < 2 or len(b) < 2:
        return {"label": label, "error": "サンプル不足"}
    ks_stat, p_val = sps.ks_2samp(a, b)
    return {
        "label": label, "n_a": len(a), "n_b": len(b),
        "ks_statistic": round(float(ks_stat), 4), "p_value": round(float(p_val), 4),
        "distributions_differ_at_5pct": bool(p_val < 0.05),
    }


# ======================================================================
# 3. Opportunity Cost分析（既存datasetの整形出力）
# ======================================================================
def opportunity_cost_summary() -> dict:
    d = _load("opportunity_cost_dataset.json")
    return {
        "adopted_baseline": d["adopted_baseline"],
        "missed_overall": d["missed_candidates_forward20_overall"],
        "by_sector": d["opportunity_cost_by_sector"],
        "by_regime": d["opportunity_cost_by_regime"],
        "by_rank": d["opportunity_cost_by_rank"],
        "by_skip_reason": d["opportunity_cost_by_skip_reason"],
    }


# ======================================================================
# 4. Sector集中分析（permutation test）
# ======================================================================
def sector_concentration_test(n_iter: int = 10000, seed: int = 42) -> dict:
    """観測されたセクター集中度が偶然（母集団のセクター分布に基づくランダム配分）よりも
    高いかをpermutation testで検定。

    ⚠ 設計上の注意（Study80Aで判明した落とし穴）: 帰無仮説の候補セクター母集団には
    「その日観測されたdistinct_sectors数」ではなく、母集団全体（missed_candidates_full.json
    の実際のsector頻度分布）を使うこと。前者を使うと、少数候補日(n=2-3)ほど必然的に
    「1セクターしかない」ケースが増え、帰無仮説自体が高集中に偏る循環論法になる。
    """
    corr = _load("correlation_dataset.json")
    missed = _load("missed_candidates_full.json")["missed_candidates"]
    observed = corr["avg_max_sector_concentration_pct"]
    daily = corr["daily_records"]
    if not daily or not missed:
        return {"error": "データなし"}

    # 母集団のセクター頻度分布（実データから構築・観測日に依存しない）
    sector_pool = [c.get("sector", "不明") for c in missed]
    sectors_unique = sorted(set(sector_pool))
    sector_freq = np.array([sector_pool.count(s) for s in sectors_unique], dtype=float)
    sector_prob = sector_freq / sector_freq.sum()

    n_cands_pool = [r["n_candidates"] for r in daily]
    rng = np.random.default_rng(seed)
    perm_concentrations = []
    for _ in range(n_iter):
        sim_shares = []
        for n_cand in n_cands_pool:
            if n_cand <= 0:
                continue
            assign = rng.choice(len(sectors_unique), size=n_cand, p=sector_prob)
            counts = np.bincount(assign)
            sim_shares.append(counts.max() / n_cand)
        if sim_shares:
            perm_concentrations.append(float(np.mean(sim_shares)) * 100)
    perm_arr = np.array(perm_concentrations)
    p_val = float(np.mean(perm_arr >= observed))
    return {
        "observed_avg_max_sector_concentration_pct": observed,
        "permutation_mean_pct(母集団分布ベース)": round(float(np.mean(perm_arr)), 2),
        "permutation_p95_pct": round(float(np.percentile(perm_arr, 95)), 2),
        "p_value_observed_gte_random": round(p_val, 4),
        "concentration_exceeds_random_at_5pct": bool(p_val < 0.05),
    }


# ======================================================================
# 5. Portfolio相関分析（同日候補相関の要約）
# ======================================================================
def portfolio_correlation_check() -> dict:
    corr = _load("correlation_dataset.json")
    return {
        "n_multi_candidate_days": corr["n_multi_candidate_days"],
        "avg_max_sector_concentration_pct": corr["avg_max_sector_concentration_pct"],
        "avg_momentum_same_direction_pct": corr["avg_momentum_same_direction_pct"],
        "interpretation": "同方向モメンタム率が高いほど、同日の複数候補は真に独立なリスクではなく"
                          "相関の高い集中ポジションである可能性が高い。50%超で方向性の偏りを示唆、"
                          "80%超は強い偏り。",
    }


# ======================================================================
# 6. 時間分散仮説（bootstrap比較: 同日グループ vs 日をまたぐIID）
# ======================================================================
def time_diversification_test(n_iter: int = 10000, seed: int = 42) -> dict:
    """「同時保有を増やすと分散が効く」仮説の検証。

    ⚠ 設計上の注意: 単純に母集団全体からIIDで3件を無作為抽出して分散を見るだけでは、
    3件が「同日に競合していたか」という真の同時保有の相関構造を反映しない
    （日をまたいだ無作為抽出は定義上ほぼ独立になり、理想的な1/3への分散縮小に近づいて当然）。
    本関数は「同日に実際に競合していた候補群」からの抽出(同日群)と、
    「日をまたいだ無作為抽出」(IID群)の両方を計算し、両者の分散縮小率を比較する。
    同日群の分散縮小がIID群より明確に小さければ、見せかけの分散(同時保有=独立ではない)を裏付ける。
    """
    fwd = _load("forward_return_dataset.json")
    corr = _load("correlation_dataset.json")
    records = {(r["date"], r.get("symbol")): r["forward_20"] for r in fwd["records"] if r.get("forward_20") is not None}
    all_vals = np.array(list(records.values()))
    if len(all_vals) < 10:
        return {"error": "サンプル不足"}

    # 同日群: correlation_dataset.jsonのdaily_recordsから3候補以上ある日を抽出し、
    # forward_return_datasetと同日付でjoinして実際の値を取得
    same_day_groups = []
    fwd_by_date: dict[str, list] = {}
    for r in fwd["records"]:
        if r.get("forward_20") is not None:
            fwd_by_date.setdefault(r["date"], []).append(r["forward_20"])
    for d, vals in fwd_by_date.items():
        if len(vals) >= 3:
            same_day_groups.append(vals[:3])

    rng = np.random.default_rng(seed)
    if same_day_groups:
        same_day_means = [float(np.mean(rng.choice(g, size=3, replace=True))) for g in same_day_groups for _ in range(max(1, n_iter // len(same_day_groups)))]
    else:
        same_day_means = []

    iid_means = [float(np.mean(rng.choice(all_vals, size=3, replace=True))) for _ in range(n_iter)]
    single_draws = rng.choice(all_vals, size=n_iter, replace=True)

    single_var = float(np.var(single_draws))
    iid_var = float(np.var(np.array(iid_means)))
    same_day_var = float(np.var(np.array(same_day_means))) if same_day_means else None

    return {
        "n_same_day_groups(3candidate+)": len(same_day_groups),
        "single_candidate_variance": round(single_var, 3),
        "iid_cross_day_3candidate_variance_reduction_pct": round((1 - iid_var / single_var) * 100, 1) if single_var else None,
        "same_day_3candidate_variance_reduction_pct": round((1 - same_day_var / single_var) * 100, 1) if same_day_var and single_var else None,
        "interpretation": "同日群の分散縮小率がIID群(日をまたぐ無作為抽出、理論値≈67%)より明確に小さければ、"
                          "同日に競合する候補同士は互いに独立ではなく、見かけの分散が実質的な集中である"
                          "ことを裏付ける。同水準ならば同時保有は真に独立なリスク分散として機能している。",
    }


# ======================================================================
# 7. 同日競合分析（rank0見送り率のbootstrap CI）
# ======================================================================
def same_day_competition_test(n_iter: int = 10000, seed: int = 42) -> dict:
    missed = _load("missed_candidates_full.json")["missed_candidates"]
    ranks = [c["rank"] for c in missed if c.get("rank") is not None]
    if not ranks:
        return {"error": "データなし"}
    arr = np.array(ranks)
    rank0_flag = (arr == 0).astype(float)
    rng = np.random.default_rng(seed)
    boots = [float(np.mean(rng.choice(rank0_flag, size=len(rank0_flag), replace=True))) * 100 for _ in range(n_iter)]
    boots_arr = np.array(boots)
    return {
        "observed_rank0_share_pct": round(float(np.mean(rank0_flag)) * 100, 1),
        "bootstrap_ci_5pct": round(float(np.percentile(boots_arr, 5)), 1),
        "bootstrap_ci_95pct": round(float(np.percentile(boots_arr, 95)), 1),
        "interpretation": "rank0(最上位候補)見送り率のCIが50%を大きく上回れば、"
                          "見送りの過半数が「質の低い候補の除外」ではなく「最良機会の喪失」であることを"
                          "統計的に裏付ける。",
    }


def main():
    print("=" * 80)
    print("  Study81 Analysis Template — 全解析を一括実行（新規BTなし）")
    print("=" * 80)

    trade_v2 = _load("trade_dataset_v2.json")["trades"]
    missed_full = _load("missed_candidates_full.json")["missed_candidates"]
    fwd = _load("forward_return_dataset.json")["records"]

    print("\n[1] 採用 vs 見送り比較（RSR・Mann-Whitney U）")
    adopted_rsr = [t["entry_rsr"] for t in trade_v2]
    missed_rsr = [c["rsr"] for c in missed_full]
    print(compare_adopted_vs_missed(adopted_rsr, missed_rsr, label="RSR"))

    print("\n[2] Forward Return比較（CAP_MISS vs SECTOR_CAP・KS検定）")
    cap_miss_fwd = [r["forward_20"] for r in fwd if r["skip_reason"] == "CAP_MISS"]
    sector_cap_fwd = [r["forward_20"] for r in fwd if r["skip_reason"] == "SECTOR_CAP"]
    print(forward_return_comparison(cap_miss_fwd, sector_cap_fwd, label="CAP_MISS vs SECTOR_CAP forward_20"))

    print("\n[3] Opportunity Cost分析")
    print(json.dumps(opportunity_cost_summary(), ensure_ascii=False, indent=1))

    print("\n[4] Sector集中分析（permutation test）")
    print(sector_concentration_test())

    print("\n[5] Portfolio相関分析")
    print(portfolio_correlation_check())

    print("\n[6] 時間分散仮説（bootstrap）")
    print(time_diversification_test())

    print("\n[7] 同日競合分析（rank0見送り率bootstrap CI）")
    print(same_day_competition_test())


if __name__ == "__main__":
    main()
