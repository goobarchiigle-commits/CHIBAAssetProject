"""
src/backtest/study103_portfolio_feasibility.py
Study103 — Portfolio Architecture Feasibility (CP2 / Research Continuation Gate)

正典:
  reports/roadmap_v15_governance_layer.md   §1/§1A/§4/§8（目的・CP2後決定木・成果物7点）
  reports/study103_design.md                §3仮定表 / §4シナリオ / §9A Sweep / §9B成果物 / §9C実装仕様

Primary Objective  : Attempt to falsify ambitious routes.
Secondary Objective: Determine feasible ceilings and terminal states.
Failure to falsify != proof.

固定事項（§9C・変更禁止）:
  - シナリオ = {Conservative, Base, Optimistic} × {CaseA(Core included), CaseB(Core excluded)} の6のみ
  - 仮定表・相関行列・Vol = §3/§9C凍結値。実行後の追加・変更は Study103B 採番 + full rerun のみ
  - 乱数モデル = 月次log-return多変量正規・5年(60M)・seed固定（正規尾部=楽観側→RED判定はa fortiori有効）
  - 配分グリッド = 5%刻み・long-only・レバ1.0x固定・SG weight<=20%（Route D overlay上限）
  - 判定統計量 = median CAGR / median Calmar / median MaxDD / RoR=P(MaxDD>30%)
  - 出力はadvisoryのみ・自動採用なし・append-only
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

try:
    from src.paths import REPORTS_DIR, RESULTS_DIR
except ImportError:  # 直接実行時のfallback（パス直書きはしない）
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.paths import REPORTS_DIR, RESULTS_DIR

RUN_DATE = "2026-07-20"
OUT_JSON = RESULTS_DIR / f"study103_portfolio_feasibility_{RUN_DATE}.json"

SLEEVES = ["Core", "MN", "PEAD", "TSMOM", "SG"]
LEVELS = ["Conservative", "Base", "Optimistic"]
HORIZON_MONTHS = 60
YEARS = 5.0
SEED_SCAN, SEED_FINAL = 42, 4242
N_SCAN, N_FINAL = 3000, 20000
WEIGHT_STEP = 0.05
SG_CAP = 0.20
TOP_K_REFINE = 40          # 各(case, calmar水準)ごとの精査候補数
CALMAR_LEVELS = [1.0, 1.3, 1.5]
TIERS = {  # v15§2 昇順体系（Tier0=Market Return=自明成立につきread-out対象外）
    "Tier3": {"cagr": 0.30, "calmar": 1.5},
    "Tier2": {"cagr": 0.20, "calmar": 1.3},
    "Tier1": {"cagr": 0.10, "calmar": 1.0},
}
DD_LIMIT, ROR_DD, ROR_LIMIT = 0.20, 0.30, 0.01

# ---- 凍結仮定（study103_design.md §3 / §9C(b)） ----
CAGR = {
    "Core":  {"Conservative": 0.00, "Base": 0.03, "Optimistic": 0.05},
    "MN":    {"Conservative": 0.12, "Base": 0.18, "Optimistic": 0.25},
    "PEAD":  {"Conservative": 0.20, "Base": 0.275, "Optimistic": 0.35},
    "TSMOM": {"Conservative": 0.15, "Base": 0.18, "Optimistic": 0.25},
    "SG":    {"Conservative": 0.20, "Base": 0.275, "Optimistic": 0.35},
}
VOL = {
    "Core":  {"Conservative": 0.18, "Base": 0.15, "Optimistic": 0.13},
    "MN":    {"Conservative": 0.14, "Base": 0.11, "Optimistic": 0.09},
    "PEAD":  {"Conservative": 0.22, "Base": 0.18, "Optimistic": 0.15},
    "TSMOM": {"Conservative": 0.25, "Base": 0.225, "Optimistic": 0.20},
    "SG":    {"Conservative": 0.35, "Base": 0.30, "Optimistic": 0.26},
}
MAXDD_ASSUMED = {  # 仮定Calmar（自動RED境界評価用）
    "Core":  {"Conservative": 0.25, "Base": 0.18, "Optimistic": 0.14},
    "MN":    {"Conservative": 0.15, "Base": 0.10, "Optimistic": 0.06},
    "PEAD":  {"Conservative": 0.20, "Base": 0.14, "Optimistic": 0.10},
    "TSMOM": {"Conservative": 0.20, "Base": 0.15, "Optimistic": 0.12},
    "SG":    {"Conservative": 0.50, "Base": 0.40, "Optimistic": 0.30},
}
PAIR_CORR = {  # §9C(a)
    ("Core", "MN"):    {"Conservative": 0.10, "Base": 0.00, "Optimistic": -0.15},
    ("Core", "PEAD"):  {"Conservative": 0.20, "Base": 0.10, "Optimistic": 0.00},
    ("Core", "TSMOM"): {"Conservative": 0.35, "Base": 0.25, "Optimistic": 0.15},
    ("Core", "SG"):    {"Conservative": 0.70, "Base": 0.55, "Optimistic": 0.40},
    ("MN", "PEAD"):    {"Conservative": 0.10, "Base": 0.05, "Optimistic": 0.00},
    ("MN", "TSMOM"):   {"Conservative": 0.10, "Base": 0.05, "Optimistic": 0.00},
    ("MN", "SG"):      {"Conservative": 0.15, "Base": 0.05, "Optimistic": 0.00},
    ("PEAD", "TSMOM"): {"Conservative": 0.25, "Base": 0.15, "Optimistic": 0.05},
    ("PEAD", "SG"):    {"Conservative": 0.45, "Base": 0.35, "Optimistic": 0.25},
    ("TSMOM", "SG"):   {"Conservative": 0.35, "Base": 0.25, "Optimistic": 0.15},
}
AUTO_RED = {"sleeves_min": 5, "avg_corr_max": 0.10, "avg_calmar_min": 2.0}  # Tier3のみ・§9C(e)


def corr_matrix(level: str) -> tuple[np.ndarray, bool]:
    n = len(SLEEVES)
    m = np.eye(n)
    for (a, b), v in PAIR_CORR.items():
        i, j = SLEEVES.index(a), SLEEVES.index(b)
        m[i, j] = m[j, i] = v[level]
    eig = np.linalg.eigvalsh(m)
    clipped = False
    if eig.min() < 1e-10:  # 非PSD→固有値クリップ+再正規化（§9C(a)）
        w, v = np.linalg.eigh(m)
        m = v @ np.diag(np.clip(w, 1e-10, None)) @ v.T
        d = np.sqrt(np.diag(m))
        m = m / np.outer(d, d)
        clipped = True
    return m, clipped


def simulate_sleeves(level: str, n_paths: int, seed: int) -> np.ndarray:
    """月次arithmetic growth factor (1+r) を返す。shape=(n_paths, 60, 5) float32"""
    corr, _ = corr_matrix(level)
    mu = np.array([np.log(1 + CAGR[s][level]) / 12.0 for s in SLEEVES])
    sg = np.array([VOL[s][level] / np.sqrt(12.0) for s in SLEEVES])
    chol = np.linalg.cholesky(corr)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths, HORIZON_MONTHS, len(SLEEVES)))
    logret = mu + (z @ chol.T) * sg
    return np.exp(logret).astype(np.float32)  # growth factors


def build_grid() -> np.ndarray:
    """5%刻み・sum=1・SG<=20% の全配分。shape=(n_w, 5)"""
    units = int(round(1 / WEIGHT_STEP))
    grid = []
    for c in itertools.combinations(range(units + len(SLEEVES) - 1), len(SLEEVES) - 1):
        # stars and bars
        parts, prev = [], -1
        for x in c:
            parts.append(x - prev - 1)
            prev = x
        parts.append(units + len(SLEEVES) - 2 - prev)
        w = np.array(parts) * WEIGHT_STEP
        if w[SLEEVES.index("SG")] <= SG_CAP + 1e-9:
            grid.append(w)
    return np.array(grid, dtype=np.float32)


def path_metrics(gf: np.ndarray, weights: np.ndarray, chunk: int = 200) -> dict[str, np.ndarray]:
    """各配分のmedCAGR/medMaxDD/medCalmar/RoR/Tier1パス合格率。gf=(P,60,5), weights=(W,5)"""
    n_w = weights.shape[0]
    out = {k: np.empty(n_w) for k in ("med_cagr", "med_maxdd", "med_calmar", "ror", "tier1_pass")}
    r_arith = gf - 1.0
    for s in range(0, n_w, chunk):
        w = weights[s:s + chunk]                                # (C,5)
        rp = np.einsum("ptk,ck->cpt", r_arith, w)               # (C,P,60)
        wealth = np.cumprod(1.0 + rp, axis=2)
        peak = np.maximum.accumulate(wealth, axis=2)
        maxdd = (1.0 - wealth / peak).max(axis=2)               # (C,P)
        cagr = wealth[:, :, -1] ** (1.0 / YEARS) - 1.0
        calmar = np.where(maxdd > 1e-9, cagr / maxdd, np.sign(cagr) * 100.0)
        t1 = (cagr >= TIERS["Tier1"]["cagr"]) & (maxdd <= DD_LIMIT) & (calmar >= TIERS["Tier1"]["calmar"])
        sl = slice(s, s + w.shape[0])
        out["med_cagr"][sl] = np.median(cagr, axis=1)
        out["med_maxdd"][sl] = np.median(maxdd, axis=1)
        out["med_calmar"][sl] = np.median(calmar, axis=1)
        out["ror"][sl] = (maxdd > ROR_DD).mean(axis=1)
        out["tier1_pass"][sl] = t1.mean(axis=1)
    return out


def feasible_mask(m: dict, cagr_req: float, calmar_req: float) -> np.ndarray:
    return ((m["med_cagr"] >= cagr_req) & (m["med_calmar"] >= calmar_req)
            & (m["med_maxdd"] <= DD_LIMIT) & (m["ror"] < ROR_LIMIT))


def weighted_avg_pair_corr(w: np.ndarray, corr: np.ndarray) -> float:
    num = den = 0.0
    for i in range(len(SLEEVES)):
        for j in range(i + 1, len(SLEEVES)):
            num += w[i] * w[j] * corr[i, j]
            den += w[i] * w[j]
    return float(num / den) if den > 1e-12 else 0.0


def weighted_assumed_calmar(w: np.ndarray, level: str) -> float:
    return float(sum(w[k] * CAGR[s][level] / MAXDD_ASSUMED[s][level] for k, s in enumerate(SLEEVES)))


def main() -> None:
    t0 = time.time()
    print("Study103 — Portfolio Architecture Feasibility (CP2)")
    print("Primary: Attempt to falsify ambitious routes. Failure to falsify != proof.\n")

    grid_all = build_grid()
    core_idx = SLEEVES.index("Core")
    cases = {"CaseA": np.ones(len(grid_all), bool), "CaseB": grid_all[:, core_idx] < 1e-9}
    print(f"配分グリッド: {len(grid_all)}（CaseB該当 {int(cases['CaseB'].sum())}）")

    results: dict = {"levels": {}, "config": {
        "sleeves": SLEEVES, "cagr": CAGR, "vol": VOL, "maxdd_assumed": MAXDD_ASSUMED,
        "pair_corr": {f"{a}-{b}": v for (a, b), v in PAIR_CORR.items()},
        "n_scan": N_SCAN, "n_final": N_FINAL, "seeds": [SEED_SCAN, SEED_FINAL],
        "horizon_months": HORIZON_MONTHS, "weight_step": WEIGHT_STEP, "sg_cap": SG_CAP,
        "dd_limit": DD_LIMIT, "ror_limit": ROR_LIMIT, "tiers": TIERS, "auto_red": AUTO_RED,
        "model": "monthly lognormal MVN (thin tails => optimistic side => RED is a fortiori valid)",
    }}

    for level in LEVELS:
        corr, clipped = corr_matrix(level)
        gf_scan = simulate_sleeves(level, N_SCAN, SEED_SCAN)
        m_scan = path_metrics(gf_scan, grid_all)
        # 精査候補: 各(case×calmar水準)で制約充足の上位 + 制約無視のCAGR上位（frontier近傍保険）
        cand = set()
        for cname, cmask in cases.items():
            for c in CALMAR_LEVELS:
                ok = feasible_mask(m_scan, 0.0, c) & cmask
                idx = np.where(ok)[0]
                idx = idx[np.argsort(-m_scan["med_cagr"][idx])][:TOP_K_REFINE]
                cand.update(idx.tolist())
                near = np.where(cmask)[0]
                near = near[np.argsort(-(m_scan["med_calmar"][near] * 0 + m_scan["med_cagr"][near]))][:TOP_K_REFINE]
                cand.update(near.tolist())
            t1_idx = np.where(cmask)[0]
            t1_idx = t1_idx[np.argsort(-m_scan["tier1_pass"][t1_idx])][:TOP_K_REFINE]
            cand.update(t1_idx.tolist())
        cand_idx = np.array(sorted(cand))
        gf_fin = simulate_sleeves(level, N_FINAL, SEED_FINAL)
        m_fin = path_metrics(gf_fin, grid_all[cand_idx], chunk=50)
        print(f"[{level}] scan={len(grid_all)} refine={len(cand_idx)} corr_clipped={clipped} "
              f"({time.time()-t0:.0f}s)")

        lv: dict = {"corr_clipped": clipped, "cases": {}}
        for cname, cmask in cases.items():
            sub = np.where(cmask[cand_idx])[0]
            frontier = {}
            for c in CALMAR_LEVELS:
                ok = sub[feasible_mask({k: v[sub] for k, v in m_fin.items()}, 0.0, c)]
                if len(ok):
                    best = ok[np.argmax(m_fin["med_cagr"][ok])]
                    gi = cand_idx[best]
                    frontier[str(c)] = {
                        "max_med_cagr": round(float(m_fin["med_cagr"][best]), 4),
                        "weights": {s: round(float(grid_all[gi, k]), 2) for k, s in enumerate(SLEEVES)
                                    if grid_all[gi, k] > 1e-9},
                        "med_maxdd": round(float(m_fin["med_maxdd"][best]), 4),
                        "med_calmar": round(float(m_fin["med_calmar"][best]), 3),
                        "ror": round(float(m_fin["ror"][best]), 5),
                    }
                else:
                    frontier[str(c)] = None
            tiers = {}
            for tname, req in TIERS.items():
                f = frontier[str(req["calmar"])]
                tiers[tname] = bool(f is not None and f["max_med_cagr"] >= req["cagr"])
            # Tier3自動RED境界（成立時のみ評価）
            auto_red_eval = None
            if tiers["Tier3"]:
                ok3 = sub[feasible_mask({k: v[sub] for k, v in m_fin.items()},
                                        TIERS["Tier3"]["cagr"], TIERS["Tier3"]["calmar"])]
                gis = cand_idx[ok3]
                sleeves_cnt = [(grid_all[g] > 1e-9).sum() for g in gis]
                corrs = [weighted_avg_pair_corr(grid_all[g], corr) for g in gis]
                calmars = [weighted_assumed_calmar(grid_all[g], level) for g in gis]
                auto_red_eval = {
                    "required_sleeves_min": int(min(sleeves_cnt)),
                    "required_avg_corr_min": round(min(corrs), 3),
                    "required_avg_assumed_calmar_min": round(min(calmars), 3),
                    "triggered": bool(min(sleeves_cnt) >= AUTO_RED["sleeves_min"]
                                      or min(corrs) < AUTO_RED["avg_corr_max"]
                                      or min(calmars) > AUTO_RED["avg_calmar_min"]),
                }
            best_t1 = sub[np.argmax(m_fin["tier1_pass"][sub])]
            lv["cases"][cname] = {
                "frontier": frontier, "tier_feasible": tiers, "auto_red_tier3": auto_red_eval,
                "best_tier1_pass_rate": round(float(m_fin["tier1_pass"][best_t1]), 4),
                "best_tier1_weights": {s: round(float(grid_all[cand_idx[best_t1], k]), 2)
                                       for k, s in enumerate(SLEEVES)
                                       if grid_all[cand_idx[best_t1], k] > 1e-9},
            }
        results["levels"][level] = lv

    # ---- Verdicts（§9C(e): GREEN=Base成立 / YELLOW=Optのみ / RED=Optでも不成立 or 自動RED） ----
    verdicts = {}
    for tname in TIERS:
        v = {}
        for cname in cases:
            feas = {lvl: results["levels"][lvl]["cases"][cname]["tier_feasible"][tname] for lvl in LEVELS}
            ar = results["levels"]["Base"]["cases"][cname]["auto_red_tier3"] if tname == "Tier3" else None
            if feas["Base"]:
                verdict = "GREEN"
                if tname == "Tier3" and ar and ar["triggered"]:
                    verdict = "RED(auto)"
            elif feas["Optimistic"]:
                verdict = "YELLOW"
                if tname == "Tier3":
                    aro = results["levels"]["Optimistic"]["cases"][cname]["auto_red_tier3"]
                    if aro and aro["triggered"]:
                        verdict = "RED(auto)"
            else:
                verdict = "RED"
            v[cname] = verdict
        verdicts[tname] = v
    verdicts["Tier0"] = {c: "GREEN (trivially feasible: market return)" for c in cases}

    # ---- Termination Probability（v15§8定義） ----
    term = np.mean([1.0 - results["levels"][lvl]["cases"][c]["best_tier1_pass_rate"]
                    for lvl in LEVELS for c in cases])
    # ---- Core Retirement Probability（Case B frontier(c=1.3) >= Case A の水準割合） ----
    def fr(lvl, c):  # noqa: E306
        f = results["levels"][lvl]["cases"][c]["frontier"]["1.3"]
        return f["max_med_cagr"] if f else -1.0
    core_ret = np.mean([1.0 if fr(lvl, "CaseB") >= fr(lvl, "CaseA") else 0.0 for lvl in LEVELS])

    results["verdicts"] = verdicts
    results["termination_probability"] = round(float(term), 4)
    results["core_retirement_probability"] = round(float(core_ret), 4)
    results["meta"] = {"run_at": datetime.now(timezone.utc).isoformat(),
                       "elapsed_sec": round(time.time() - t0, 1)}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- サマリ ----
    print("\n===== Goal Ladder Sweep (verdicts) =====")
    for tname in ["Tier3", "Tier2", "Tier1", "Tier0"]:
        print(f"  {tname}: " + " / ".join(f"{c}={v}" for c, v in verdicts[tname].items()))
    print("\n===== Goal frontier (max median CAGR @ Calmar constraint, Base scenario) =====")
    for cname in cases:
        for c in CALMAR_LEVELS:
            f = results["levels"]["Base"]["cases"][cname]["frontier"][str(c)]
            s = (f"CAGR {f['max_med_cagr']:+.1%} w={f['weights']} DD {f['med_maxdd']:.1%} "
                 f"RoR {f['ror']:.2%}") if f else "infeasible"
            print(f"  [{cname}] Calmar>={c}: {s}")
    print(f"\nTermination Probability   : {term:.1%}")
    print(f"Core Retirement Probability: {core_ret:.1%}")
    print(f"JSON: {OUT_JSON}")
    print(f"elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
