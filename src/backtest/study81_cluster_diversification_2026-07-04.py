"""
study81_cluster_diversification_2026-07-04.py
Study81 — Cluster Diversification Hypothesis

仮説: 「max_positions=3が最適」なのではなく「4銘柄目は既存3銘柄と同じクラスターに
属するため期待値が増えない」。

**追加BT禁止**。Study80AのResearch Assets（trade_dataset_v2.json/
missed_candidates_full.json/forward_return_dataset.json/opportunity_cost_dataset.json/
correlation_dataset.json）のみを使用。新規BT・run_scenario呼び出しは一切行わない。

Cluster ID設計（工夫）:
  - macro_cluster: 既存Production計装 src/strategy/cluster.py の CLUSTER_MAP_DEFAULT
    （cyclical_macro/defensive/growth_tech/real_asset/other）をそのまま再利用。
    これは新規発明ではなく、composite_alpha_bt.py が CLUSTER_CAP 判定に実際に使っている
    既存の risk grouping であり、「クラスター」概念として最も直接的に妥当。
  - factor_cluster: momentum_63d_pct・atr_pct・rsrの結合z-scoreによるtercile(Low/Mid/High)。
    alpha_scoreはStudy80Aで判明の通り全件ほぼ0（degenerate）のためクラスタリングに使用しない
    （観測は残すが、次元としては採用しない旨を明記）。
  - cluster_id = f"{macro_cluster}|{factor_cluster}"
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from src.strategy.cluster import CLUSTER_MAP_DEFAULT, get_cluster

ROOT = Path(__file__).resolve().parents[2]
BT_DIR = ROOT / "backtests"
TODAY_STR = "2026-07-04"


def load(name: str) -> dict:
    with open(BT_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def build_symbol_sector_map(trade_v2: list[dict], missed: list[dict]) -> dict[str, str]:
    m = {}
    for t in trade_v2:
        m[t["symbol"]] = t.get("entry_sector") or t.get("sector", "不明")
    for c in missed:
        m.setdefault(c["symbol"], c.get("sector", "不明"))
    return m


def factor_tercile(values: list[float], target: float | None) -> str:
    if target is None or not values:
        return "unknown"
    p33, p67 = np.percentile(values, 33), np.percentile(values, 67)
    if target <= p33:
        return "Low"
    if target <= p67:
        return "Mid"
    return "High"


def main():
    print("=" * 80)
    print("  Study81 — Cluster Diversification Hypothesis（追加BT禁止・既存Assetsのみ）")
    print(f"  Date: {TODAY_STR}")
    print("=" * 80)

    trade_v2 = load("trade_dataset_v2.json")["trades"]
    missed_data = load("missed_candidates_full.json")
    missed = missed_data["missed_candidates"]
    opp_cost = load("opportunity_cost_dataset.json")
    corr = load("correlation_dataset.json")

    print(f"\n[LOAD] 採用トレード{len(trade_v2)}件 / 見送り候補{len(missed)}件（新規BTなし）")

    sym_sector = build_symbol_sector_map(trade_v2, missed)

    # ── 解析1: Cluster化 ──────────────────────────────────────────────────
    print("\n[解析1] Cluster ID生成（macro_cluster×factor_cluster）...")
    mom_pool = [c.get("momentum_63d_pct") for c in missed if c.get("momentum_63d_pct") is not None]
    atr_pool = [c.get("atr_pct") for c in missed if c.get("atr_pct") is not None] + \
               [t.get("entry_atr_pct") for t in trade_v2 if t.get("entry_atr_pct") is not None]
    rsr_pool = [c.get("rsr") for c in missed if c.get("rsr") is not None] + \
               [t.get("entry_rsr") for t in trade_v2 if t.get("entry_rsr") is not None]

    def assign_cluster(symbol, momentum, atr, rsr, alpha=None):
        macro = get_cluster(symbol, sym_sector, CLUSTER_MAP_DEFAULT)
        mom_bin = factor_tercile(mom_pool, momentum)
        atr_bin = factor_tercile(atr_pool, atr)
        rsr_bin = factor_tercile(rsr_pool, rsr)
        factor_cluster = f"mom{mom_bin[0]}_atr{atr_bin[0]}_rsr{rsr_bin[0]}"
        return macro, factor_cluster, f"{macro}|{factor_cluster}"

    cluster_records = []
    for t in trade_v2:
        macro, fc, cid = assign_cluster(t["symbol"], None, t.get("entry_atr_pct"), t.get("entry_rsr"))
        cluster_records.append({
            "source": "adopted", "symbol": t["symbol"], "date": t["entry_date"],
            "macro_cluster": macro, "factor_cluster": fc, "cluster_id": cid,
            "sector": sym_sector.get(t["symbol"], "不明"),
            "pnl": t["pnl"], "return_pct": t["return_pct"],
        })
    for c in missed:
        macro, fc, cid = assign_cluster(c["symbol"], c.get("momentum_63d_pct"), c.get("atr_pct"), c.get("rsr"))
        cluster_records.append({
            "source": "missed", "symbol": c["symbol"], "date": c["date"], "skip_reason": c.get("skip_reason"),
            "macro_cluster": macro, "factor_cluster": fc, "cluster_id": cid,
            "sector": c.get("sector", "不明"), "rank": c.get("rank"),
            "forward_20": c.get("forward_20"), "mfe_pct": c.get("mfe_pct"), "mae_pct": c.get("mae_pct"),
            "selected_symbols": c.get("selected_symbols", []),
        })
    macro_counts = {}
    for r in cluster_records:
        macro_counts[r["macro_cluster"]] = macro_counts.get(r["macro_cluster"], 0) + 1
    print(f"  macro_cluster分布: {macro_counts}")
    print("  ⚠ alpha_scoreは全件ほぼ0(degenerate)のためクラスタリング次元から除外（Study80A確認済み）")

    # ── 解析2: Cluster別 勝率/PF/Expectancy/MFE/MAE/Forward Return ─────────
    print("\n[解析2] Cluster別統計...")
    cluster_stats = {}
    for macro in list(CLUSTER_MAP_DEFAULT.keys()) + ["other"]:
        adopted_in = [r for r in cluster_records if r["source"] == "adopted" and r["macro_cluster"] == macro]
        missed_in = [r for r in cluster_records if r["source"] == "missed" and r["macro_cluster"] == macro]
        if not adopted_in and not missed_in:
            continue
        winners = [r for r in adopted_in if r["pnl"] >= 0]
        losers = [r for r in adopted_in if r["pnl"] < 0]
        win_rate = round(100 * len(winners) / max(1, len(adopted_in)), 1) if adopted_in else None
        gp = sum(r["pnl"] for r in winners); gl = abs(sum(r["pnl"] for r in losers))
        pf = round(gp / gl, 3) if gl > 0 else (float("inf") if gp > 0 else None)
        expectancy = round(float(np.mean([r["pnl"] for r in adopted_in])), 1) if adopted_in else None
        missed_fwd20 = [r["forward_20"] for r in missed_in if r.get("forward_20") is not None]
        missed_mfe = [r["mfe_pct"] for r in missed_in if r.get("mfe_pct") is not None]
        missed_mae = [r["mae_pct"] for r in missed_in if r.get("mae_pct") is not None]
        cluster_stats[macro] = {
            "n_adopted": len(adopted_in), "n_missed": len(missed_in),
            "win_rate_pct": win_rate, "profit_factor": pf, "expectancy_yen": expectancy,
            "missed_mean_forward_20_pct": round(float(np.mean(missed_fwd20)), 2) if missed_fwd20 else None,
            "missed_mean_mfe_pct": round(float(np.mean(missed_mfe)), 2) if missed_mfe else None,
            "missed_mean_mae_pct": round(float(np.mean(missed_mae)), 2) if missed_mae else None,
        }
        print(f"  {macro}: adopted={len(adopted_in)}(WR={win_rate}%,PF={pf},Exp={expectancy}円) "
              f"missed={len(missed_in)}(fwd20={cluster_stats[macro]['missed_mean_forward_20_pct']}%)")

    # ── 解析3: Portfolio内Cluster集中率（実測 vs ランダム） ─────────────────
    print("\n[解析3] Portfolio内Cluster集中率（実測 vs ランダム）...")
    portfolio_days = []
    for c in missed:
        held = c.get("selected_symbols", [])
        if len(held) < 2:
            continue
        held_clusters = [get_cluster(s, sym_sector, CLUSTER_MAP_DEFAULT) for s in held]
        counts = {}
        for hc in held_clusters:
            counts[hc] = counts.get(hc, 0) + 1
        max_share = max(counts.values()) / len(held_clusters)
        portfolio_days.append({"date": c["date"], "held": held, "clusters": held_clusters,
                                 "max_cluster_concentration_pct": round(max_share * 100, 1)})
    # 重複日付を除去（同日複数missed候補があるため）
    seen_dates = set()
    unique_portfolio_days = []
    for p in portfolio_days:
        if p["date"] not in seen_dates:
            seen_dates.add(p["date"])
            unique_portfolio_days.append(p)

    observed_conc = round(float(np.mean([p["max_cluster_concentration_pct"] for p in unique_portfolio_days])), 1) if unique_portfolio_days else None

    # ランダム帰無仮説: 母集団のmacro_cluster頻度分布からランダム配分
    all_macro_pool = [r["macro_cluster"] for r in cluster_records]
    macro_unique = sorted(set(all_macro_pool))
    macro_prob = np.array([all_macro_pool.count(m) for m in macro_unique], dtype=float)
    macro_prob = macro_prob / macro_prob.sum()
    rng = np.random.default_rng(42)
    n_iter = 10000
    perm_concs = []
    held_sizes = [len(p["held"]) for p in unique_portfolio_days]
    for _ in range(n_iter):
        sims = []
        for n_held in held_sizes:
            if n_held <= 0:
                continue
            assign = rng.choice(len(macro_unique), size=n_held, p=macro_prob)
            counts = np.bincount(assign)
            sims.append(counts.max() / n_held)
        if sims:
            perm_concs.append(float(np.mean(sims)) * 100)
    perm_arr = np.array(perm_concs)
    p_value = float(np.mean(perm_arr >= observed_conc)) if observed_conc is not None else None
    portfolio_cluster_report = {
        "n_portfolio_snapshots": len(unique_portfolio_days),
        "observed_avg_max_cluster_concentration_pct": observed_conc,
        "random_null_mean_pct": round(float(np.mean(perm_arr)), 2) if len(perm_arr) else None,
        "random_null_p95_pct": round(float(np.percentile(perm_arr, 95)), 2) if len(perm_arr) else None,
        "p_value_observed_gte_random": round(p_value, 4) if p_value is not None else None,
        "concentration_exceeds_random_at_5pct": bool(p_value < 0.05) if p_value is not None else None,
        "daily_snapshots_sample": unique_portfolio_days[:20],
    }
    print(f"  実測平均クラスター集中度: {observed_conc}%  ランダム帰無仮説平均: {portfolio_cluster_report['random_null_mean_pct']}%  "
          f"p値: {portfolio_cluster_report['p_value_observed_gte_random']}")

    # ── 解析4: 4銘柄目が同クラスター vs 別クラスターでForward Return比較 ────
    print("\n[解析4] 4銘柄目(CAP_MISS候補) 同クラスターvs別クラスター Forward Return比較...")
    same_cluster_fwd, diff_cluster_fwd = [], []
    for c in missed:
        if c.get("skip_reason") != "CAP_MISS" or c.get("forward_20") is None:
            continue
        held = c.get("selected_symbols", [])
        if not held:
            continue
        cand_cluster = get_cluster(c["symbol"], sym_sector, CLUSTER_MAP_DEFAULT)
        held_clusters = {get_cluster(s, sym_sector, CLUSTER_MAP_DEFAULT) for s in held}
        if cand_cluster in held_clusters:
            same_cluster_fwd.append(c["forward_20"])
        else:
            diff_cluster_fwd.append(c["forward_20"])

    from scipy import stats as sps
    same_arr, diff_arr = np.array(same_cluster_fwd), np.array(diff_cluster_fwd)
    if len(same_arr) >= 2 and len(diff_arr) >= 2:
        u_stat, p_val_mw = sps.mannwhitneyu(same_arr, diff_arr, alternative="two-sided")
    else:
        u_stat, p_val_mw = None, None
    analysis4 = {
        "n_same_cluster": len(same_arr), "n_diff_cluster": len(diff_arr),
        "same_cluster_mean_forward_20_pct": round(float(np.mean(same_arr)), 2) if len(same_arr) else None,
        "same_cluster_median_forward_20_pct": round(float(np.median(same_arr)), 2) if len(same_arr) else None,
        "diff_cluster_mean_forward_20_pct": round(float(np.mean(diff_arr)), 2) if len(diff_arr) else None,
        "diff_cluster_median_forward_20_pct": round(float(np.median(diff_arr)), 2) if len(diff_arr) else None,
        "mannwhitney_u": round(float(u_stat), 2) if u_stat is not None else None,
        "p_value": round(float(p_val_mw), 4) if p_val_mw is not None else None,
        "significant_at_5pct": bool(p_val_mw < 0.05) if p_val_mw is not None else None,
    }
    print(f"  同クラスター(n={len(same_arr)}): mean_fwd20={analysis4['same_cluster_mean_forward_20_pct']}%")
    print(f"  別クラスター(n={len(diff_arr)}): mean_fwd20={analysis4['diff_cluster_mean_forward_20_pct']}%")
    print(f"  Mann-Whitney U p値: {analysis4['p_value']}")

    # ── 解析5: Opportunity CostをCluster単位で集計 ──────────────────────────
    print("\n[解析5] Opportunity Cost（Cluster単位）...")
    opp_by_cluster = {}
    for macro in list(CLUSTER_MAP_DEFAULT.keys()) + ["other"]:
        vals = [r["forward_20"] for r in cluster_records if r["source"] == "missed" and r["macro_cluster"] == macro and r.get("forward_20") is not None]
        if vals:
            opp_by_cluster[macro] = {"n": len(vals), "mean_forward_20_pct": round(float(np.mean(vals)), 2),
                                       "median_forward_20_pct": round(float(np.median(vals)), 2)}
    print(f"  Cluster別Opportunity Cost: {opp_by_cluster}")

    # ── 解析6: Hidden Factor探索（Clusterで説明できないケースのみ抽出） ──────
    print("\n[解析6] Hidden Factor探索（Cluster仮説で説明できないケース抽出）...")
    hidden_cases = []
    for c in missed:
        if c.get("skip_reason") != "CAP_MISS" or c.get("forward_20") is None:
            continue
        held = c.get("selected_symbols", [])
        if not held:
            continue
        cand_cluster = get_cluster(c["symbol"], sym_sector, CLUSTER_MAP_DEFAULT)
        held_clusters = {get_cluster(s, sym_sector, CLUSTER_MAP_DEFAULT) for s in held}
        is_diff_cluster = cand_cluster not in held_clusters
        # Cluster仮説の予測: 別クラスターなら「真に分散効果があり見送りは機会損失」のはず
        # →しかしforward_20が大きく負（-10%以下）なら、Cluster仮説だけでは説明できない
        #   (別クラスターなのに結果が悪い＝クラスター以外の要因が支配的な可能性)
        if is_diff_cluster and c["forward_20"] is not None and c["forward_20"] < -10.0:
            hidden_cases.append({
                "date": c["date"], "symbol": c["symbol"], "rank": c.get("rank"),
                "cand_cluster": cand_cluster, "held_clusters": list(held_clusters),
                "forward_20": c["forward_20"], "mae_pct": c.get("mae_pct"),
                "rsr": c.get("rsr"), "atr_pct": c.get("atr_pct"), "momentum_63d_pct": c.get("momentum_63d_pct"),
                "market_regime": c.get("market_regime"),
                "note": "別クラスターだが結果は大幅マイナス — クラスター理論だけでは説明不可",
            })
        # 逆パターン: 同クラスターなのに大幅プラス（同クラスターは伸びないはずという予測に反する）
        if not is_diff_cluster and c["forward_20"] is not None and c["forward_20"] > 15.0:
            hidden_cases.append({
                "date": c["date"], "symbol": c["symbol"], "rank": c.get("rank"),
                "cand_cluster": cand_cluster, "held_clusters": list(held_clusters),
                "forward_20": c["forward_20"], "mae_pct": c.get("mae_pct"),
                "rsr": c.get("rsr"), "atr_pct": c.get("atr_pct"), "momentum_63d_pct": c.get("momentum_63d_pct"),
                "market_regime": c.get("market_regime"),
                "note": "同クラスターだが大幅プラス — クラスター理論の予測(伸びないはず)に反する",
            })
    print(f"  Cluster理論で説明できないケース: {len(hidden_cases)}件")

    # ── 保存 ─────────────────────────────────────────────────────────────
    print("\n[SAVE] 成果物保存中...")
    with open(BT_DIR / "cluster_dataset.json", "w", encoding="utf-8") as f:
        json.dump({"date": TODAY_STR, "n_records": len(cluster_records), "cluster_map": CLUSTER_MAP_DEFAULT,
                    "records": cluster_records}, f, ensure_ascii=False, indent=2, default=str)
    with open(BT_DIR / "cluster_statistics.json", "w", encoding="utf-8") as f:
        json.dump({"date": TODAY_STR, "cluster_stats": cluster_stats, "analysis4_same_vs_diff_cluster": analysis4,
                    "opportunity_cost_by_cluster": opp_by_cluster}, f, ensure_ascii=False, indent=2, default=str)
    with open(BT_DIR / "portfolio_cluster_report.json", "w", encoding="utf-8") as f:
        json.dump(portfolio_cluster_report, f, ensure_ascii=False, indent=2, default=str)
    with open(BT_DIR / "hidden_cases.json", "w", encoding="utf-8") as f:
        json.dump({"date": TODAY_STR, "n_hidden_cases": len(hidden_cases), "cases": hidden_cases},
                   f, ensure_ascii=False, indent=2, default=str)

    print("  [OUTPUT] backtests/cluster_dataset.json")
    print("  [OUTPUT] backtests/cluster_statistics.json")
    print("  [OUTPUT] backtests/portfolio_cluster_report.json")
    print("  [OUTPUT] backtests/hidden_cases.json")

    return {
        "cluster_stats": cluster_stats, "portfolio_cluster_report": portfolio_cluster_report,
        "analysis4": analysis4, "hidden_cases_count": len(hidden_cases),
    }


if __name__ == "__main__":
    main()
