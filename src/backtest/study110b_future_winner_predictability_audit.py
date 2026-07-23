"""
src/backtest/study110b_future_winner_predictability_audit.py
Study110B — Future Winner Predictability Audit（記述統計 + permutation検定・Study110A panel再利用のみ）

正典: reports/study110a_future_winner_definition_audit.md（Overlap=Case2境界・Persistence=弱いlift）
実装承認: ユーザー"Study110B"タスク指示（2026-07-22）

目的（狭く固定）:
  Future winnerは予測可能な対象か？ H0: mostly stochastic（Universe研究自体が無意味）
  vs H1: conditional persistence exists（sector/regime/state依存）を判定する。

方法: Study110Aで既に保存済みのpanel（fwd_3M/6M/12M）をそのまま再利用し、新規計算しない。
  Decile遷移行列（10x10・月次クロスセクション内qcut）を構築し、
  Tier1=Transition Entropy・Permutation Null(N=1000)・Economic Lift Thresholdで主判定、
  Tier2=Sector/Bull-Bear/State persistenceを診断専用として算出する。

禁止事項（厳守）: alpha探索・ML・新factor・新データ取得・backtestは一切行わない。
  Study110A panel再利用のみ（新規forward return計算なし）。

データ源: backtests/study110a_panel_enriched_2026-07-22.csv（既存・Study110A出力）
          database/market/master/companies.parquet（sector・Study95と同一ソース）
          data/jquants/processed/TOPIX.parquet（regime・Study95と同一ソース）

Decision rule（ユーザー指定・事前固定）: if p>=0.05 or lift<1.5: TERMINAL else: Study112へ進行
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
PANEL_CSV = RESULTS_DIR / "study110a_panel_enriched_2026-07-22.csv"
OUT_JSON = RESULTS_DIR / f"study110b_future_winner_predictability_audit_{RUN_DATE}.json"
OUT_MD = REPORTS_DIR / "study110b_future_winner_predictability_audit.md"

HORIZONS_MONTHS = {"3M": 3, "6M": 6, "12M": 12}
N_DECILES = 10
MIN_CROSS_SECTION_N = 30
N_PERMUTATIONS = 1000
PERM_SEED = 42
LIFT_THRESHOLD = 1.5
P_THRESHOLD = 0.05
MAX_ENTROPY_BITS = float(np.log2(N_DECILES))


# ---------------------------------------------------------------- decile割当（月次CS内）
def assign_decile_col(panel: pd.DataFrame, value_col: str, out_col: str) -> pd.DataFrame:
    panel[out_col] = np.nan
    for rb_date, idx in panel.groupby("rebalance_date").groups.items():
        sub = panel.loc[idx, value_col].dropna()
        if len(sub) < MIN_CROSS_SECTION_N:
            continue
        ranks = sub.rank(method="first")
        deciles = pd.qcut(ranks, N_DECILES, labels=False) + 1
        panel.loc[sub.index, out_col] = deciles
    return panel


# ---------------------------------------------------------------- transition統計（entropy・lift）
def transition_stats(decile_t: np.ndarray, decile_th: np.ndarray) -> tuple[float, float, np.ndarray]:
    idx = (decile_t.astype(int) - 1) * N_DECILES + (decile_th.astype(int) - 1)
    counts = np.bincount(idx, minlength=N_DECILES * N_DECILES).reshape(N_DECILES, N_DECILES).astype(float)
    row_sums = counts.sum(axis=1)
    valid_rows = row_sums > 0
    probs = np.zeros_like(counts)
    probs[valid_rows] = counts[valid_rows] / row_sums[valid_rows, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        row_entropy = -np.nansum(np.where(probs > 0, probs * np.log2(probs), 0.0), axis=1)
    mean_entropy = float(np.average(row_entropy[valid_rows], weights=row_sums[valid_rows]))
    entropy_ratio = mean_entropy / MAX_ENTROPY_BITS
    diag_probs = np.diagonal(probs)
    mean_diag = float(np.average(diag_probs[valid_rows], weights=row_sums[valid_rows]))
    lift = mean_diag / (1.0 / N_DECILES)
    return entropy_ratio, lift, counts


def build_blocks(decile_t: np.ndarray, decile_th: np.ndarray, block_ids: np.ndarray):
    order = np.argsort(block_ids, kind="stable")
    sorted_block_ids = block_ids[order]
    sorted_t = decile_t[order]
    sorted_th = decile_th[order]
    _, start_idx = np.unique(sorted_block_ids, return_index=True)
    start_idx = np.append(np.sort(start_idx), len(sorted_block_ids))
    start_idx = np.unique(start_idx)
    return sorted_t, sorted_th, start_idx


def permutation_test(decile_t: np.ndarray, decile_th: np.ndarray, block_ids: np.ndarray,
                      n_perm: int = N_PERMUTATIONS, seed: int = PERM_SEED) -> dict:
    obs_entropy_ratio, obs_lift, _ = transition_stats(decile_t, decile_th)
    sorted_t, sorted_th, bounds = build_blocks(decile_t, decile_th, block_ids)
    rng = np.random.default_rng(seed)
    null_entropy = np.empty(n_perm)
    null_lift = np.empty(n_perm)
    work = sorted_th.copy()
    for p in range(n_perm):
        for i in range(len(bounds) - 1):
            s, e = bounds[i], bounds[i + 1]
            work[s:e] = rng.permutation(sorted_th[s:e])
        er, lf, _ = transition_stats(sorted_t, work)
        null_entropy[p] = er
        null_lift[p] = lf
    p_entropy = float((1 + np.sum(null_entropy <= obs_entropy_ratio)) / (1 + n_perm))
    p_lift = float((1 + np.sum(null_lift >= obs_lift)) / (1 + n_perm))
    return {
        "observed_entropy_ratio": round(obs_entropy_ratio, 4),
        "observed_lift": round(obs_lift, 4),
        "null_entropy_ratio_mean": round(float(null_entropy.mean()), 4),
        "null_entropy_ratio_std": round(float(null_entropy.std()), 4),
        "null_lift_mean": round(float(null_lift.mean()), 4),
        "null_lift_std": round(float(null_lift.std()), 4),
        "p_value_entropy": round(p_entropy, 4),
        "p_value_lift": round(p_lift, 4),
        "n_permutations": n_perm,
    }


def build_pairs(panel: pd.DataFrame, decile_col: str, months: int, universe_keys: set[str]) -> pd.DataFrame:
    """(rebalance_date, code, decile_t, decile_t+H) を全date-pairで結合。block_id=起点date。"""
    by_date = {rb: g.set_index("code")[decile_col] for rb, g in panel.groupby("rebalance_date")}
    rows = []
    for rb_str in sorted(by_date.keys()):
        t0 = pd.Timestamp(rb_str)
        t1_key = (t0 + pd.DateOffset(months=months)).strftime("%Y-%m-01")
        if t1_key not in universe_keys or t1_key not in by_date:
            continue
        s0 = by_date[rb_str].dropna()
        s1 = by_date[t1_key].dropna()
        common = s0.index.intersection(s1.index)
        if len(common) < MIN_CROSS_SECTION_N:
            continue
        rows.append(pd.DataFrame({
            "block_id": rb_str, "code": common,
            "decile_t": s0.loc[common].to_numpy(), "decile_th": s1.loc[common].to_numpy(),
        }))
    if not rows:
        return pd.DataFrame(columns=["block_id", "code", "decile_t", "decile_th"])
    return pd.concat(rows, ignore_index=True)


def tier1_for_horizon(panel: pd.DataFrame, horizon: str, months: int, universe_keys: set[str]) -> dict:
    decile_col = f"decile_{horizon}"
    pairs = build_pairs(panel, decile_col, months, universe_keys)
    if pairs.empty:
        return {"note": "insufficient pairs"}
    block_ids_numeric = pd.factorize(pairs["block_id"])[0]
    stats = permutation_test(pairs["decile_t"].to_numpy(), pairs["decile_th"].to_numpy(), block_ids_numeric)
    stats["n_pairs"] = int(len(pairs))
    stats["n_blocks"] = int(pairs["block_id"].nunique())
    verdict = "TERMINAL" if (stats["p_value_lift"] >= P_THRESHOLD or stats["observed_lift"] < LIFT_THRESHOLD) \
        else "PROCEED_TO_STUDY112"
    stats["verdict"] = verdict
    stats["decision_rule"] = f"if p>={P_THRESHOLD} or lift<{LIFT_THRESHOLD}: TERMINAL else: PROCEED_TO_STUDY112"
    return stats, pairs


def tier2_sector(pairs: pd.DataFrame, panel: pd.DataFrame, sector_map: pd.DataFrame) -> dict:
    code_sector = sector_map["Sector17CodeName"]
    pairs = pairs.copy()
    pairs["sector"] = pairs["code"].map(code_sector)
    out = {}
    for sector, g in pairs.dropna(subset=["sector"]).groupby("sector"):
        if len(g) < MIN_CROSS_SECTION_N:
            out[sector] = {"n": int(len(g)), "note": "insufficient_n"}
            continue
        er, lf, _ = transition_stats(g["decile_t"].to_numpy(), g["decile_th"].to_numpy())
        out[sector] = {"n": int(len(g)), "entropy_ratio": round(er, 4), "lift": round(lf, 4)}
    return out


def tier2_regime(pairs: pd.DataFrame, regime_at_date: dict[str, str]) -> dict:
    pairs = pairs.copy()
    pairs["regime"] = pairs["block_id"].map(regime_at_date)
    out = {}
    for regime_v in ("Above200MA", "Below200MA"):
        g = pairs[pairs["regime"] == regime_v]
        if len(g) < MIN_CROSS_SECTION_N:
            out[regime_v] = {"n": int(len(g)), "note": "insufficient_n"}
            continue
        er, lf, _ = transition_stats(g["decile_t"].to_numpy(), g["decile_th"].to_numpy())
        out[regime_v] = {"n": int(len(g)), "entropy_ratio": round(er, 4), "lift": round(lf, 4)}
    return out


def tier2_state(pairs: pd.DataFrame) -> dict:
    er, lf, counts = transition_stats(pairs["decile_t"].to_numpy(), pairs["decile_th"].to_numpy())
    row_sums = counts.sum(axis=1)
    probs = np.divide(counts, row_sums[:, None], out=np.zeros_like(counts), where=row_sums[:, None] > 0)
    q10_row = probs[9]  # decile10 = index9
    q1_row = probs[0]
    return {
        "entropy_ratio": round(er, 4), "lift": round(lf, 4),
        "Q10_to_Q10": round(float(q10_row[9]), 4),
        "Q10_to_Q8plus": round(float(q10_row[7] + q10_row[8] + q10_row[9]), 4),
        "Q1_to_Q1": round(float(q1_row[0]), 4),
        "Q1_to_Q3minus": round(float(q1_row[0] + q1_row[1] + q1_row[2]), 4),
        "full_transition_matrix_probs": probs.round(4).tolist(),
    }


def main() -> None:
    print("Study110B — Future Winner Predictability Audit（Study110A panel再利用のみ・新規計算ゼロ）")

    print("[1/5] Study110A panel読込...")
    panel = pd.read_csv(PANEL_CSV, encoding="utf-8")
    panel["rebalance_date"] = panel["rebalance_date"].astype(str)
    print(f"  panel rows={len(panel):,}")

    print("[2/5] Universe C keys / Sector map / TOPIX regime読込（Study95と同一関数）...")
    monthly_universe = s95.load_universe()
    universe_keys = set(monthly_universe.keys())
    sector_map = s95.load_sector_map()
    topix = s95.load_topix_calendar()
    topix_sma200 = topix.rolling(s95.REGIME_MA_PERIOD, min_periods=s95.REGIME_MA_PERIOD).mean()
    regime_bull_series = topix >= topix_sma200
    regime_at_date: dict[str, str] = {}
    for rb_str in panel["rebalance_date"].unique():
        t0 = pd.Timestamp(rb_str)
        at_or_before = regime_bull_series.index[regime_bull_series.index <= t0]
        if len(at_or_before) == 0:
            continue
        v = regime_bull_series.loc[at_or_before[-1]]
        regime_at_date[rb_str] = "Above200MA" if bool(v) else "Below200MA"

    print("[3/5] Decile割当（3horizon×月次クロスセクション内qcut）...")
    for h in HORIZONS_MONTHS:
        panel = assign_decile_col(panel, f"fwd_{h}", f"decile_{h}")

    print("[4/5] Tier1: Transition Entropy + Permutation Null(N=1000) + Lift（horizon別）...")
    result: dict = {"run_at": datetime.now(timezone.utc).isoformat(), "n_panel_rows": int(len(panel)),
                     "params": {"n_deciles": N_DECILES, "n_permutations": N_PERMUTATIONS,
                                "lift_threshold": LIFT_THRESHOLD, "p_threshold": P_THRESHOLD}}
    tier1 = {}
    pairs_by_horizon = {}
    for h, months in HORIZONS_MONTHS.items():
        print(f"  horizon={h} ...")
        stats, pairs = tier1_for_horizon(panel, h, months, universe_keys)
        tier1[h] = stats
        pairs_by_horizon[h] = pairs
    result["tier1_transition_entropy_permutation_lift"] = tier1

    verdicts = [tier1[h]["verdict"] for h in HORIZONS_MONTHS if "verdict" in tier1[h]]
    any_proceed = any(v == "PROCEED_TO_STUDY112" for v in verdicts)
    all_terminal = all(v == "TERMINAL" for v in verdicts) if verdicts else True
    result["overall_verdict"] = "TERMINAL" if all_terminal else (
        "PROCEED_TO_STUDY112_CONDITIONAL" if any_proceed else "MIXED")

    print("[5/5] Tier2: Sector / Bull-Bear / State persistence（診断専用）...")
    tier2 = {}
    for h in HORIZONS_MONTHS:
        pairs = pairs_by_horizon[h]
        if pairs.empty:
            tier2[h] = {"note": "insufficient pairs"}
            continue
        tier2[h] = {
            "sector_persistence": tier2_sector(pairs, panel, sector_map),
            "regime_persistence": tier2_regime(pairs, regime_at_date),
            "state_persistence": tier2_state(pairs),
        }
    result["tier2_diagnostic"] = tier2

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\nJSON: {OUT_JSON}")
    print(f"Overall verdict: {result['overall_verdict']}")


if __name__ == "__main__":
    main()
