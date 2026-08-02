"""
src/backtest/study110b_sector_conditioned_formalization.py
Study110B Sector-Conditioned Persistence Formalization（正式follow-up・新規Study番号なし）

正典: ユーザータスク指示"Study110B Sector-Conditioned Persistence Formalization"
      （2026-08-02・permutation null修正版・ASK_FIRST承認済み）

目的（狭く固定）:
  Study110B Tier2で診断専用に留めた「sector-conditioned subset persistence」
  （Sector17別lift 0.44-1.97・permutation検定なし・n=68-3546）を、Tier1と同一の
  厳密さで正式検定する。「Sectorという粗い経済的制約で絞るだけでwinner persistenceが
  market-wide baseline(lift 1.2-1.3)より強くなるか」の単一仮説を、17 sector全てを
  事前固定で集約した1つの統計量でのみ判定する（個別sectorのcherry-pick一切禁止）。

研究仮説（事前固定）:
  H: 事前固定された同一Sector17分類内に限定したwinner subsetは、
     全市場winner subsetよりも持続性が高い。
  本結果はEconomic Causal Graphそのものの検証ではない（Sector17は粗いproxy）。

Permutation null設計（Study110B Tier1からの唯一の変更点・ユーザー指示で修正済み）:
  Tier1は block=rebalance_date単位でdecile_thをshuffleし、全pairsをpoolして
  単一lift統計量を計算する。これは「sector rotation効果」（Study98/99/111で既に
  REJECT）と「sector内の個別銘柄相対持続性」（本Studyが検証したい対象）を混同する。
  本Studyでは block=(rebalance_date × Sector17)単位でdecile_thをshuffleし、
  各permutation replicateについてもsector別transition_statsをn加重平均する
  ことで、sector構成・sector別銘柄数・sectorの実現decile分布を観測値と
  null両方で保持したまま、「sector内での個別銘柄の相対持続性」だけを検定する。

再利用（import・無改変）: src.backtest.study110b_future_winner_predictability_audit
  assign_decile_col / transition_stats / build_pairs / N_DECILES / MIN_CROSS_SECTION_N
  / N_PERMUTATIONS / PERM_SEED / LIFT_THRESHOLD / P_THRESHOLD / HORIZONS_MONTHS / PANEL_CSV
src.backtest.study95_cs_momentum_factor_level.load_sector_map（Sector17CodeName・
  companies.parquetベース・Study110B Tier2と同一ソース・同一非PIT制約を継承）
src.backtest.study95_cs_momentum_factor_level.load_universe（monthly_universe keys）

新規処理（Study110Bに存在しないもの）:
  1. block_id=(rebalance_date, Sector17CodeName)単位のdecile_th shuffle
     （build_blocksと同一のsort+境界ロジックをsector配列にも適用する3配列版）
  2. sector-weighted-average lift = Σ(sector_n × sector_lift) / Σ(sector_n)
     （観測値・null replicate 1000回とも同一の集約方法・特定sector選択なし）
  3. Discovery(2016-01-01〜2022-12-31) / Validation(2023-01-01〜2026-07-31)分割
     （project既存慣行・起点rebalance_date基準・Study111と同一境界）

判定基準（Study110Bから無変更）: LIFT_THRESHOLD=1.5 / P_THRESHOLD=0.05 /
  N_PERMUTATIONS=1000 / PERM_SEED=42。discovery/validation両方がPROCEED条件を
  満たす場合のみ"支持候補"（ユーザー指定の判定表§6）。

禁止事項（厳守）: alpha探索・新feature・新threshold・Sector33への変更・horizon追加・
  sector単位でのcherry-pick・production/live/Scheduler/PARAMS_LOCKED変更・
  新規Study番号・新規データ取得、いずれも一切行わない。

出力: backtests/study110b_sector_conditioned_formalization.json（1個のみ）
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
    from src.paths import RESULTS_DIR
    import src.backtest.study95_cs_momentum_factor_level as s95
    from src.backtest.study110b_future_winner_predictability_audit import (
        HORIZONS_MONTHS, LIFT_THRESHOLD, MIN_CROSS_SECTION_N, N_PERMUTATIONS,
        PERM_SEED, P_THRESHOLD, assign_decile_col, build_pairs,
        transition_stats,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.paths import RESULTS_DIR
    import src.backtest.study95_cs_momentum_factor_level as s95
    from src.backtest.study110b_future_winner_predictability_audit import (
        HORIZONS_MONTHS, LIFT_THRESHOLD, MIN_CROSS_SECTION_N, N_PERMUTATIONS,
        PERM_SEED, P_THRESHOLD, assign_decile_col, build_pairs,
        transition_stats,
    )

RUN_DATE = "2026-08-02"
# study110b_future_winner_predictability_audit.PANEL_CSV は backtests/ 直下を指すが、
# 未コミットの既存reports再編（本Study着手前から作業ツリーに存在・本Studyとは無関係）で
# 実体は research/studies/Study110/ へ移動済み。study110bファイル自体は無改変のまま、
# 本スクリプト側でのみ実在パスを直接指定する（新規データ生成ではなく既存ファイルの参照のみ）。
PANEL_CSV = Path(__file__).resolve().parents[2] / "research" / "studies" / "Study110" / "study110a_panel_enriched_2026-07-22.csv"
OUT_JSON = RESULTS_DIR / f"study110b_sector_conditioned_formalization_{RUN_DATE}.json"

DISCOVERY = ("2016-01-01", "2022-12-31")  # project既存慣行（Study111と同一境界）
VALIDATION = ("2023-01-01", "2026-07-31")


# ======================================================================
# 1. sector配列も含めた3配列版sort（build_blocksと同一ロジック・sector追加のみ）
# ======================================================================
def _sorted_by_block(decile_t: np.ndarray, decile_th: np.ndarray, sector: np.ndarray,
                      block_ids: np.ndarray):
    order = np.argsort(block_ids, kind="stable")
    sorted_block_ids = block_ids[order]
    sorted_t = decile_t[order]
    sorted_th = decile_th[order]
    sorted_sector = sector[order]
    _, start_idx = np.unique(sorted_block_ids, return_index=True)
    start_idx = np.append(np.sort(start_idx), len(sorted_block_ids))
    start_idx = np.unique(start_idx)
    return sorted_t, sorted_th, sorted_sector, start_idx


# ======================================================================
# 2. sector-weighted-average lift（観測値・null replicate共通の集約関数）
# ======================================================================
def _sector_weighted_lift(decile_t: np.ndarray, decile_th: np.ndarray, sector: np.ndarray,
                           min_n: int) -> tuple[float | None, float | None, dict]:
    """sector別transition_statsをn加重平均する。min_n未満のsectorは除外（weight=0）。
    戻り値: (weighted_lift, weighted_entropy, per_sector_diagnostics)"""
    per_sector: dict[str, dict] = {}
    total_n = 0
    lift_sum = 0.0
    entropy_sum = 0.0
    for sec in np.unique(sector):
        mask = sector == sec
        n = int(mask.sum())
        if n < min_n:
            per_sector[str(sec)] = {"n": n, "note": "insufficient_n"}
            continue
        er, lf, _ = transition_stats(decile_t[mask], decile_th[mask])
        per_sector[str(sec)] = {"n": n, "entropy_ratio": round(er, 4), "lift": round(lf, 4)}
        total_n += n
        lift_sum += lf * n
        entropy_sum += er * n
    if total_n == 0:
        return None, None, per_sector
    return lift_sum / total_n, entropy_sum / total_n, per_sector


# ======================================================================
# 3. sector内shuffle permutation検定（Tier1 permutation_testのsector版）
# ======================================================================
def sector_conditioned_permutation_test(decile_t: np.ndarray, decile_th: np.ndarray,
                                         sector: np.ndarray, date_sector_block_ids: np.ndarray,
                                         min_n_sector: int, n_perm: int = N_PERMUTATIONS,
                                         seed: int = PERM_SEED) -> dict:
    obs_lift, obs_entropy, per_sector_obs = _sector_weighted_lift(decile_t, decile_th, sector, min_n_sector)
    if obs_lift is None:
        return {"note": "no sector has n>=min_n_sector", "observed_weighted_lift": None}

    sorted_t, sorted_th, sorted_sector, bounds = _sorted_by_block(
        decile_t, decile_th, sector, date_sector_block_ids)
    rng = np.random.default_rng(seed)
    null_lift = np.empty(n_perm)
    null_entropy = np.empty(n_perm)
    work = sorted_th.copy()
    for p in range(n_perm):
        for i in range(len(bounds) - 1):
            s, e = bounds[i], bounds[i + 1]
            work[s:e] = rng.permutation(sorted_th[s:e])
        lf, er, _ = _sector_weighted_lift(sorted_t, work, sorted_sector, min_n_sector)
        null_lift[p] = lf if lf is not None else np.nan
        null_entropy[p] = er if er is not None else np.nan

    valid = ~np.isnan(null_lift)
    p_lift = float((1 + np.sum(null_lift[valid] >= obs_lift)) / (1 + valid.sum()))
    lifts_included = [v["lift"] for v in per_sector_obs.values() if "lift" in v]
    verdict = "TERMINAL" if (p_lift >= P_THRESHOLD or obs_lift < LIFT_THRESHOLD) else "PROCEED"
    return {
        "observed_weighted_lift": round(obs_lift, 4),
        "observed_weighted_entropy_ratio": round(obs_entropy, 4),
        "null_lift_mean": round(float(np.nanmean(null_lift)), 4),
        "null_lift_std": round(float(np.nanstd(null_lift)), 4),
        "p_value_lift": round(p_lift, 4),
        "n_sectors_included": len(lifts_included),
        "median_sector_lift_diagnostic_only": round(float(np.median(lifts_included)), 4) if lifts_included else None,
        "n_permutations": int(valid.sum()),
        "verdict": verdict,
        "decision_rule": f"if p>={P_THRESHOLD} or weighted_lift<{LIFT_THRESHOLD}: TERMINAL else: PROCEED",
        "per_sector_diagnostic_not_used_for_verdict": per_sector_obs,
    }


def _period_mask(block_dates: pd.Series, start: str, end: str) -> np.ndarray:
    d = pd.to_datetime(block_dates)
    return ((d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))).to_numpy()


def main() -> int:
    print("Study110B Sector-Conditioned Persistence Formalization")

    print("[1/5] Study110A panel読込・decile割当（Study110Bと同一関数）...")
    panel = pd.read_csv(PANEL_CSV, encoding="utf-8")
    panel["rebalance_date"] = panel["rebalance_date"].astype(str)
    for h in HORIZONS_MONTHS:
        panel = assign_decile_col(panel, f"fwd_{h}", f"decile_{h}")

    print("[2/5] Universe C keys / Sector17 map読込（Study95/110Bと同一関数）...")
    # s95.UNIVERSE_FILEもbacktests/直下を指すが、上記PANEL_CSVと同一理由（無関係な
    # 既存reorgで実体がresearch/studies/Study075/へ移動済み）でruntimeパス補正のみ行う。
    # study95ファイル自体は無改変。
    _actual_universe_file = Path(__file__).resolve().parents[2] / "research" / "studies" / "Study075" / "study75_rule_universe.json"
    if _actual_universe_file.exists():
        s95.UNIVERSE_FILE = _actual_universe_file
    monthly_universe = s95.load_universe()
    universe_keys = set(monthly_universe.keys())
    sector_map = s95.load_sector_map()["Sector17CodeName"]

    result: dict = {
        "study": "Study110B_Sector_Conditioned_Formalization",
        "title": "Study110B Tier2 sector persistenceの正式permutation検定（follow-up・新規Study番号なし）",
        "run_at": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "同一Sector17分類内に限定したwinner subsetは全市場winner subsetより持続性が高い",
        "not_a_test_of": "Economic Causal Graphそのもの（Sector17は粗いproxy・classification非PIT）",
        "params": {
            "n_deciles": 10, "n_permutations": N_PERMUTATIONS, "perm_seed": PERM_SEED,
            "lift_threshold": LIFT_THRESHOLD, "p_threshold": P_THRESHOLD,
            "min_cross_section_n": MIN_CROSS_SECTION_N,
            "discovery": DISCOVERY, "validation": VALIDATION,
            "permutation_block": "rebalance_date x Sector17CodeName（Tier1のrebalance_date単独から修正）",
            "aggregation": "sector-weighted-average lift = sum(n_sector*lift_sector)/sum(n_sector)・cherry-pick禁止",
        },
        "full_market_baseline_reference_study110b_tier1": {
            "3M": 1.3038, "6M": 1.2867, "12M": 1.2080,
            "note": "Study110B Tier1既存結果の再掲（再計算なし）",
        },
        "results": {},
    }

    for h, months in HORIZONS_MONTHS.items():
        print(f"[3/5] horizon={h} pairs構築（Study110Bと同一build_pairs）...")
        pairs = build_pairs(panel, f"decile_{h}", months, universe_keys)
        if pairs.empty:
            result["results"][h] = {"note": "insufficient pairs"}
            continue
        pairs["sector"] = pairs["code"].map(sector_map)
        pairs = pairs.dropna(subset=["sector"]).reset_index(drop=True)

        horizon_result: dict = {}
        for period_name, (start, end) in [("full", ("2016-01-01", "2026-12-31")),
                                           ("discovery", DISCOVERY), ("validation", VALIDATION)]:
            print(f"  [4/5] {h} / {period_name} permutation検定（N={N_PERMUTATIONS}）...")
            mask = _period_mask(pairs["block_id"], start, end)
            sub = pairs.loc[mask]
            if len(sub) < MIN_CROSS_SECTION_N:
                horizon_result[period_name] = {"note": f"insufficient pairs (n={len(sub)})"}
                continue
            composite_block = pd.factorize(sub["block_id"].astype(str) + "|" + sub["sector"].astype(str))[0]
            stats = sector_conditioned_permutation_test(
                sub["decile_t"].to_numpy(), sub["decile_th"].to_numpy(),
                sub["sector"].to_numpy(), composite_block, MIN_CROSS_SECTION_N)
            stats["n_pairs"] = int(len(sub))
            stats["n_blocks_date_x_sector"] = int(len(np.unique(composite_block)))
            horizon_result[period_name] = stats
            v = stats.get("verdict", "N/A")
            wl = stats.get("observed_weighted_lift")
            print(f"    weighted_lift={wl}  verdict={v}")

        disc_v = horizon_result.get("discovery", {}).get("verdict")
        val_v = horizon_result.get("validation", {}).get("verdict")
        if disc_v == "PROCEED" and val_v == "PROCEED":
            combined = "SUPPORTED_CANDIDATE"
        elif disc_v is None or val_v is None:
            combined = "MEASUREMENT_LIMITATION"
        else:
            combined = "FAIL"
        horizon_result["combined_discovery_validation_verdict"] = combined
        result["results"][h] = horizon_result

    print("[5/5] CASE A/B/C判定・出力生成...")
    combined_all = [result["results"][h].get("combined_discovery_validation_verdict")
                     for h in HORIZONS_MONTHS if h in result["results"]]
    if all(c == "SUPPORTED_CANDIDATE" for c in combined_all) and combined_all:
        case = "CASE_A"
    elif any(c == "MEASUREMENT_LIMITATION" for c in combined_all):
        case = "CASE_C"
    else:
        case = "CASE_B"
    result["final_case_classification"] = case
    result["prohibited_confirmed"] = {
        "bt_engine_used": False, "strategy_backtest": False, "new_jquants_calls": False,
        "sector33_used": False, "sector_cherry_picked": False,
        "new_study_number_assigned": False, "production_code_changed": False,
        "threshold_changed_from_study110b": False,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\nJSON: {OUT_JSON}")
    print(f"Final case classification: {case}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
